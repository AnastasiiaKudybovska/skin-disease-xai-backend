from pydantic import BaseModel, EmailStr
from datetime import date

class UserBase(BaseModel):
    first_name: str
    last_name: str
    username: str
    email: EmailStr
    date_of_birth: date

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: str

class UserLoginResponse(UserBase):
    access_token: str
    refresh_token: str