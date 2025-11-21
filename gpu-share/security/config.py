# 安全配置

# JWT配置
JWT_SECRET = "your-super-secret-jwt-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_TOKEN_EXPIRE = 3600  # 1小时（秒）

# 设备认证配置
DEVICE_AUTH_ENABLED = True
DEVICE_TOKEN_EXPIRE = 86400  # 24小时（秒）

# API密钥（生产环境应从环境变量或安全存储中获取）
API_KEY = "your-api-key-change-in-production"

# TLS配置
TLS_ENABLED = True
TLS_CERT_FILE = "certs/gpu-share.crt"
TLS_KEY_FILE = "certs/gpu-share.key"

# 设备白名单（可选）
DEVICE_WHITELIST = [
    # 示例设备ID
    "worker-001",
    "worker-002"
]

# 敏感数据加密
ENCRYPT_SENSITIVE_DATA = True
