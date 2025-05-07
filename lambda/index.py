
# lambda/index.py
import json
import os
import re
import urllib.request # Python標準ライブラリを使用
import urllib.error   # エラーハンドリング用

# FastAPIエンドポイントURL (環境変数から取得することを推奨)
# Google ColabでFastAPIを起動した際に表示されるngrokのURLに /predict (FastAPIのエンドポイント) を追加したもの
FASTAPI_ENDPOINT_URL = os.environ.get("FASTAPI_ENDPOINT_URL", "YOUR_FASTAPI_NGROK_URL/predict")

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))
        
        # Cognitoで認証されたユーザー情報を取得 (これはそのまま利用可能)
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")
        
        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        # 最初はシンプルにmessageのみを使用。発展として会話履歴を利用。
        conversation_history = body.get('conversationHistory', []) 
        
        print("Processing message:", message)

        # FastAPIへのリクエストペイロードを作成
        # アドバイスに従い、最初はmessageのみをpromptとして送信するシンプルな形を想定。
        # FastAPI側が { "message": "...", "conversationHistory": [...] } を期待する場合：
        request_payload_to_fastapi = {
            "message": message,
            "conversationHistory": conversation_history # FastAPI側の実装に合わせて調整
        }
        
        print(f"Calling FastAPI endpoint: {FASTAPI_ENDPOINT_URL} with payload:", json.dumps(request_payload_to_fastapi))
        
        # FastAPIエンドポイントを呼び出し (urllib.requestを使用)
        try:
            # リクエストデータはbytes型にエンコードする
            data = json.dumps(request_payload_to_fastapi).encode('utf-8')
            
            req = urllib.request.Request(
                FASTAPI_ENDPOINT_URL,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=30) as response: # タイムアウト設定 (秒)
                response_data_bytes = response.read()
                response_content_type = response.info().get_content_type()

                if response.status != 200: # HTTPステータスコードの確認
                    raise urllib.error.HTTPError(
                        FASTAPI_ENDPOINT_URL, response.status, 
                        f"FastAPI service returned status {response.status}",
                        response.headers, response_data_bytes
                    )

                # レスポンスを解析 (FastAPI側の InferenceResponse モデルに合わせる)
                if response_content_type == 'application/json':
                    api_response_data = json.loads(response_data_bytes.decode('utf-8'))
                else:
                    # FastAPIがJSON以外を返した場合のフォールバックやエラー処理
                    print(f"Unexpected content type: {response_content_type}")
                    raise Exception(f"FastAPI service returned non-JSON response: {response_data_bytes.decode('utf-8', errors='ignore')}")

            print("FastAPI response:", json.dumps(api_response_data, default=str))

            if not api_response_data.get("success") or "response" not in api_response_data:
                raise Exception("Invalid response from FastAPI service")

            assistant_response = api_response_data["response"]
            # FastAPIが会話履歴を更新して返す場合
            updated_conversation_history = api_response_data.get("conversationHistory", conversation_history)


        except urllib.error.HTTPError as e:
            # HTTPエラー (4xx, 5xx)
            error_body = e.read().decode('utf-8', errors='ignore') if e.fp else "No error body"
            print(f"HTTPError calling FastAPI service: {e.code} {e.reason}. Body: {error_body}")
            raise Exception(f"Failed to connect to the inference API (HTTP {e.code}): {error_body}")
        except urllib.error.URLError as e:
            # ネットワーク関連エラー (ホストが見つからない等)
            print(f"URLError calling FastAPI service: {e.reason}")
            raise Exception(f"Failed to connect to the inference API (URL Error): {e.reason}")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON response from FastAPI: {e}")
            raise Exception("Failed to parse response from the inference API.")

        
        # 成功レスポンスの返却
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*", # 必要に応じて適切なオリジンに変更
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": updated_conversation_history
            })
        }
        
    except Exception as error:
        print("Error:", str(error))
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }

