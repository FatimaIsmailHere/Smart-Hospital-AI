from sqlalchemy import create_engine
import pandas as pd

# Connection parameters
host = "gateway01.ap-southeast-1.prod.aws.tidbcloud.com"
port = 4000
user = "2AGgKDuoHGcPw2W.root"
password = "T08ddmPIqVWScIIq"   
database = "test"


try:
    print("🔄 Connecting to TiDB with SSL...")
    engine = create_engine(
        f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?ssl_verify_cert=true&ssl_verify_identity=true"
    )

    print("✅ Connection created, running test query...")
    df = pd.read_sql("SELECT NOW();", engine)

    print("📌 Query result:")
    print(df)

except Exception as e:
    print("❌ Error:", e)
