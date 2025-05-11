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

no_data_to_update_exception = HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST,
    detail="No data to update"
)

no_changes_made_exception = HTTPException(
    status_code=status.HTTP_304_NOT_MODIFIED,
    detail="No changes were made"
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

invalid_lime_image_exception = HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST, detail="LIME explanation could not be generated."
)