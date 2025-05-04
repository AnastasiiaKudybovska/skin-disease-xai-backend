from fastapi import HTTPException, status

email_already_registered_exception = HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
)

email_not_registered_exception = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail="Email not registered",
)

invalid_credentials_exception = HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email or password"
)

invalid_token_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
)

user_not_found_exception = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
)


