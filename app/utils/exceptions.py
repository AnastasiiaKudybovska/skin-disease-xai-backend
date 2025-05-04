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

user_history_not_found_exception = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND, detail="User history not found"
)

invalid_history_id_exception = HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid history ID"
)

invalid_image_id_exception = HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image ID"
)

image_not_found_exception = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND, detail="Image not found"
)