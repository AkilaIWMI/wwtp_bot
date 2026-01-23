import uvicorn
from fastapi import FastAPI, Form, Response
from twilio.twiml.messaging_response import MessagingResponse

app = FastAPI()

@app.post("/whatsapp")
async def whatsapp_reply(Body: str = Form(...)):
    # 1. GET THE VALUE
    # FastAPI automatically extracts the 'Body' form field sent by Twilio
    incoming_msg = Body.lower()
    print(f"User sent: {incoming_msg}")

    # 2. PROCESS
    response_text = f"I received your message: '{incoming_msg}'. Analysis complete."

    # 3. SEND BACK THE RESULT
    resp = MessagingResponse()
    resp.message(response_text)
    
    # Return XML response with correct media type
    return Response(content=str(resp), media_type="application/xml")

if __name__ == "__main__":
    # Running on port 5000 to match your current ngrok setup
    uvicorn.run(app, host="0.0.0.0", port=5000)