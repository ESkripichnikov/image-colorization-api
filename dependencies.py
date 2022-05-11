from fastapi import Header, HTTPException
from dotenv import load_dotenv
import os

load_dotenv()


async def verify_password(admin_password: str = Header(...)):
    if admin_password != os.getenv("ADMIN_PASSWORD"):
        raise HTTPException(status_code=403, detail="Admin-password header invalid")
