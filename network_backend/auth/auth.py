from fastapi.security import OAuth2PasswordBearer

oauth2_schema = OAuth2PasswordBearer(tokenUrl='token')

# @app.get()
# def something(token: str = Depends(oauth2_schema))