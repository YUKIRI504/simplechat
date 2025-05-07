# lambda/index.py
import json
import os
import re
import urllib.request # 標準ライブラリ
import urllib.error

# FastAPIのエンドポイントURL。環境変数 FASTAPI_ENDPOINT_URL から取得。
# 未設定の場合、Google Colab等でFastAPIを起動した際に表示されるngrokのURLに
# /predict (FastAPIのエンドポイント) を追加したものを想定。
FASTAPI_ENDPOINT_URL = os.environ.get("FASTAPI_ENDPOINT_URL", "YOUR_FASTAPI_NGROK_URL/predict")

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))

        # Cognitoで認証されたユーザー情報を取得
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")

        # リクエストボディを解析
        body = json.loads(event['body'])
        message = body['message']
        # 会話履歴も取得。FastAPI側で使用されることを想定。
        conversation_history = body.get('conversationHistory', [])

        print("Processing message:", message)

        # FastAPIへのリクエストペイロードを作成
        request_payload_to_fastapi = {
            "message": message,
            "conversationHistory": conversation_history
        }

        print(f"Calling FastAPI endpoint: {FASTAPI_ENDPOINT_URL} with payload:", json.dumps(request_payload_to_fastapi))

        # FastAPIエンドポイントを呼び出し (urllib.requestを使用)
        try:
            # リクエストデータはbytes型にエンコード
            data = json.dumps(request_payload_to_fastapi).encode('utf-8')

            req = urllib.request.Request(
                FASTAPI_ENDPOINT_URL,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                response_data_bytes = response.read()
                response_content_type = response.info().get_content_type()

                if response.status != 200: # HTTPステータスコードを確認
                    raise urllib.error.HTTPError(
                        FASTAPI_ENDPOINT_URL, response.status,
                        f"FastAPI service returned status {response.status}",
                        response.headers, response_data_bytes
                    )

                # FastAPIからのレスポンスを解析 (InferenceResponseモデルを想定)
                if response_content_type == 'application/json':
                    api_response_data = json.loads(response_data_bytes.decode('utf-8'))
                else:
                    # FastAPIがJSON形式以外でレスポンスした場合のエラーハンドリング
                    print(f"Unexpected content type: {response_content_type}")
                    raise Exception(f"FastAPI service returned non-JSON response: {response_data_bytes.decode('utf-8', errors='ignore')}")

            print("FastAPI response:", json.dumps(api_response_data, default=str))

            if not api_response_data.get("success") or "response" not in api_response_data:
                raise Exception("Invalid response from FastAPI service")

            assistant_response = api_response_data["response"]
            # FastAPI側で会話履歴が更新されて返却される場合に対応
            updated_conversation_history = api_response_data.get("conversationHistory", conversation_history)

        except urllib.error.HTTPError as e:
            # HTTPエラー (クライアントエラー 4xx, サーバーエラー 5xx)
            error_body = e.read().decode('utf-8', errors='ignore') if e.fp else "No error body"
            print(f"HTTPError calling FastAPI service: {e.code} {e.reason}. Body: {error_body}")
            raise Exception(f"Failed to connect to the inference API (HTTP {e.code}): {error_body}")
        except urllib.error.URLError as e:
            # ネットワーク関連のエラー (例: ホストが見つからない)
            print(f"URLError calling FastAPI service: {e.reason}")
            raise Exception(f"Failed to connect to the inference API (URL Error): {e.reason}")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON response from FastAPI: {e}")
            raise Exception("Failed to parse response from the inference API.")


        # 処理成功時のレスポンスを返却
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
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
