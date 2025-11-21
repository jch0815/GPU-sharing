#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TLS证书生成脚本
"""

import os
import sys
import logging
import datetime

# 导入安全模块
try:
    from .crypto import generate_self_signed_cert
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("加密模块不可用，无法生成证书")
    sys.exit(1)

logger = logging.getLogger(__name__)

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="生成TLS证书")
    parser.add_argument("--cert-dir", default="certs", help="证书目录")
    parser.add_argument("--cert-name", default="gpu-share", help="证书名称")
    parser.add_argument("--country", default="CN", help="国家代码")
    parser.add_argument("--state", default="Beijing", help="省/州")
    parser.add_argument("--locality", default="Beijing", help="城市")
    parser.add_argument("--org", default="GPU Share Platform", help="组织名称")
    parser.add_argument("--common-name", default="gpu-share", help="通用名称")
    parser.add_argument("--days", type=int, default=365, help="有效期天数")

    args = parser.parse_args()

    # 创建证书目录
    cert_dir = args.cert_dir
    os.makedirs(cert_dir, exist_ok=True)

    # 证书文件路径
    cert_file = os.path.join(cert_dir, f"{args.cert_name}.crt")
    key_file = os.path.join(cert_dir, f"{args.cert_name}.key")

    # 生成证书
    logger.info(f"生成自签名证书: {cert_file}, {key_file}")

    success = generate_self_signed_cert(
        cert_file=cert_file,
        key_file=key_file,
        country=args.country,
        state=args.state,
        locality=args.locality,
        organization=args.org,
        common_name=args.common_name
    )

    if success:
        logger.info("证书生成成功")

        # 显示证书信息
        import subprocess
        try:
            # 显示证书内容
            subprocess.run(['openssl', 'x509', '-in', cert_file, '-text', '-noout'], check=True)

            # 显示私钥内容
            subprocess.run(['openssl', 'rsa', '-in', key_file, '-text', '-noout'], check=True)

            # 验证证书
            subprocess.run(['openssl', 'verify', '-CAfile', cert_file, cert_file], check=True)
        except Exception as e:
            logger.error(f"验证证书失败: {str(e)}")
    else:
        logger.error("证书生成失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
