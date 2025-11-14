-- InsightFlow Database Schema

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    organization VARCHAR(255),
    tier VARCHAR(50) DEFAULT 'free',
    api_quota_daily INT DEFAULT 100,
    api_quota_remaining INT DEFAULT 100,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Sales data table (example business data)
CREATE TABLE IF NOT EXISTS sales_transactions (
    transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL,
    region VARCHAR(100),
    product VARCHAR(100),
    sales DECIMAL(12,2),
    cost DECIMAL(12,2),
    units INTEGER,
    customer_segment VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_date_region (date, region),
    INDEX idx_product (product),
    INDEX idx_segment (customer_segment)
);

-- Query logs for analytics
CREATE TABLE IF NOT EXISTS query_logs (
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    question TEXT NOT NULL,
    sql_generated TEXT,
    execution_time_ms FLOAT,
    status VARCHAR(50),
    error_message TEXT,
    result_rows INT,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_user_date (user_id, created_at),
    INDEX idx_status (status)
);

-- Context knowledge base
CREATE TABLE IF NOT EXISTS knowledge_base (
    kb_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100),
    tags TEXT[],
    embedding VECTOR(1536),  -- For vector similarity search
    created_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User feedback for continuous improvement
CREATE TABLE IF NOT EXISTS query_feedback (
    feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES query_logs(query_id),
    user_id UUID REFERENCES users(user_id),
    rating INT CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert sample data
INSERT INTO sales_transactions (date, region, product, sales, cost, units, customer_segment)
SELECT
    date '2024-01-01' + (i % 365) * interval '1 day' as date,
    CASE (i % 4)
        WHEN 0 THEN 'North'
        WHEN 1 THEN 'South'
        WHEN 2 THEN 'East'
        ELSE 'West'
    END as region,
    CASE (i % 3)
        WHEN 0 THEN 'Laptop'
        WHEN 1 THEN 'Monitor'
        ELSE 'Keyboard'
    END as product,
    10000 + (random() * 5000)::decimal(12,2) as sales,
    5000 + (random() * 2500)::decimal(12,2) as cost,
    (50 + random() * 50)::int as units,
    CASE (i % 3)
        WHEN 0 THEN 'Enterprise'
        WHEN 1 THEN 'SMB'
        ELSE 'Consumer'
    END as customer_segment
FROM generate_series(1, 10000) as i;

-- Create indexes for performance
CREATE INDEX idx_sales_date ON sales_transactions(date);
CREATE INDEX idx_sales_region_product ON sales_transactions(region, product);

-- Create materialized view for common aggregations (performance optimization)
CREATE MATERIALIZED VIEW sales_summary AS
SELECT
    date_trunc('day', date) as date,
    region,
    product,
    customer_segment,
    SUM(sales) as total_sales,
    SUM(cost) as total_cost,
    SUM(units) as total_units,
    COUNT(*) as transaction_count
FROM sales_transactions
GROUP BY date_trunc('day', date), region, product, customer_segment;

-- Create refresh function
CREATE OR REPLACE FUNCTION refresh_sales_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW sales_summary;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO insightflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO insightflow;
