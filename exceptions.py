# src/exceptions.py
class TextractParseError(Exception):
    pass

class S3UploadError(Exception):
    pass

class GraphExecutionError(Exception):
    pass

class DbExecutionError(Exception):
    pass
