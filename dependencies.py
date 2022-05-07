from fastapi import Header, HTTPException


async def verify_password(admin_password: str = Header(...)):
    if admin_password != "fake-super-secret-token":
        raise HTTPException(status_code=403, detail="Admin-password header invalid")
