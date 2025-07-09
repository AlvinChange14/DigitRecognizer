from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, ImageMessage, TextSendMessage
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import requests
from dotenv import load_dotenv

# 載入 .env 文件中的環境變數（本地測試用）
load_dotenv()

app = Flask(__name__)

# 從環境變數中讀取 LINE 憑證
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

# 確認憑證是否存在
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN 或 LINE_CHANNEL_SECRET 未設置，請在環境變數中配置")

# LINE Bot 配置
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 載入預訓練模型
model = load_model('temp_model/digit_recognizer.h5')

# 圖片預處理函數
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # 轉為灰階
    img = img.resize((28, 28))  # 調整大小為 28x28
    img_array = np.array(img) / 255.0  # 正規化
    img_array = img_array.reshape(1, 28, 28, 1)  # 調整為模型輸入格式 (1, 28, 28, 1)
    return img_array

# Webhook 路由
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    if signature is None:
        app.logger.error("Missing X-Line-Signature header.")
        abort(400)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.error("Invalid signature. Check your channel secret and request headers.")
        abort(400)
    except Exception as e:
        app.logger.error(f"Unhandled exception: {str(e)}")
        abort(400)

    return 'OK'

# 處理圖片訊息
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    # 取得圖片內容
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)
    
    # 儲存圖片到 static 資料夾
    os.makedirs('static', exist_ok=True)
    image_path = os.path.join('static', f'{message_id}.jpg')
    with open(image_path, 'wb') as f:
        f.write(message_content.content)
    
    # 預處理並預測
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction, axis=1)[0]
    
    # 回傳結果
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=f'辨識結果：{predicted_digit}')
    )
    
    # 刪除暫存圖片
    os.remove(image_path)

# 處理文字訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text='請傳送一張手寫數字圖片！')
    )

if __name__ == "__main__":
    app.run(debug=True)
