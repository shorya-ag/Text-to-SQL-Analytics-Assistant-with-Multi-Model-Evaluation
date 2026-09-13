create database rag_evaluation;
use rag_evaluation;

ALTER TABLE campaigns ADD PRIMARY KEY (campaign_id);
ALTER TABLE channels ADD PRIMARY KEY (channel_id);
ALTER TABLE clicks ADD PRIMARY KEY (click_id);
ALTER TABLE conversions ADD PRIMARY KEY (conversion_id);
ALTER TABLE customers ADD PRIMARY KEY (customer_id);
ALTER TABLE impressions ADD PRIMARY KEY (impression_id);
ALTER TABLE products ADD PRIMARY KEY (product_id);

-- clicks
ALTER TABLE clicks ADD CONSTRAINT fk_clicks_campaign
    FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id);
ALTER TABLE clicks ADD CONSTRAINT fk_clicks_customer
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id);
ALTER TABLE clicks ADD CONSTRAINT fk_clicks_channel
    FOREIGN KEY (channel_id) REFERENCES channels(channel_id);
ALTER TABLE clicks ADD CONSTRAINT fk_clicks_impression
    FOREIGN KEY (impression_id) REFERENCES  impressions(impression_id);

-- impressions
ALTER TABLE impressions ADD CONSTRAINT fk_impressions_campaign
    FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id);
ALTER TABLE impressions ADD CONSTRAINT fk_impressions_customer
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id);
ALTER TABLE impressions ADD CONSTRAINT fk_impressions_channel
    FOREIGN KEY (channel_id) REFERENCES channels(channel_id);

-- campaigns
ALTER TABLE campaigns ADD CONSTRAINT fk_campaigns_channel
    FOREIGN KEY (channel_id) REFERENCES channels(channel_id);
ALTER TABLE campaigns ADD CONSTRAINT fk_campaigns_product
    FOREIGN KEY (product_id) REFERENCES products(product_id);

-- conversions
ALTER TABLE conversions ADD CONSTRAINT fk_conversions_click
    FOREIGN KEY (click_id) REFERENCES clicks(click_id);
ALTER TABLE conversions ADD CONSTRAINT fk_conversions_campaign
    FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id);
ALTER TABLE conversions ADD CONSTRAINT fk_conversions_customer
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id);
ALTER TABLE conversions ADD CONSTRAINT fk_conversions_product
    FOREIGN KEY (product_id) REFERENCES products(product_id);
    
SELECT
    TABLE_NAME,
    COLUMN_NAME,
    CONSTRAINT_NAME,
    REFERENCED_TABLE_NAME,
    REFERENCED_COLUMN_NAME
FROM information_schema.KEY_COLUMN_USAGE
WHERE REFERENCED_TABLE_NAME IS NOT NULL
  AND TABLE_SCHEMA = 'rag_evaluation';
  
SELECT cl.click_id, cl.impression_id
FROM clicks cl
LEFT JOIN impressions i ON cl.impression_id = i.impression_id
WHERE i.impression_id IS NULL
  AND cl.impression_id IS NOT NULL;

UPDATE clicks cl
LEFT JOIN impressions i ON cl.impression_id = i.impression_id
SET cl.impression_id = NULL
WHERE i.impression_id IS NULL
  AND cl.impression_id IS NOT NULL;