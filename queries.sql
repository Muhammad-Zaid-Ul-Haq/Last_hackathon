-- ====================================================================
-- Data Science Final Hackathon: E-Commerce Customer Intelligence
-- Five Core Business Analysis Queries
-- ====================================================================

-- --------------------------------------------------------------------
-- QUERY 1: Total Net Revenue
-- Calculates cumulative net revenue after applying product discounts
-- --------------------------------------------------------------------
SELECT 
    ROUND(SUM(quantity * unit_price * (1.0 - discount)), 2) AS total_net_revenue,
    COUNT(order_id) AS total_orders,
    ROUND(AVG(quantity * unit_price * (1.0 - discount)), 2) AS avg_order_value
FROM orders;


-- --------------------------------------------------------------------
-- QUERY 2: Top 10 Customers by Spending
-- Identifies top spending customers with location, order volume, and revenue
-- --------------------------------------------------------------------
SELECT 
    c.customer_id,
    c.name AS customer_name,
    c.city,
    c.membership_type,
    COUNT(o.order_id) AS total_orders,
    ROUND(SUM(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS total_spending,
    ROUND(AVG(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS avg_order_value
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name, c.city, c.membership_type
ORDER BY total_spending DESC
LIMIT 10;


-- --------------------------------------------------------------------
-- QUERY 3: Category-wise Revenue, Order Count, and Quantity Sold
-- Evaluates product performance and volume across categories
-- --------------------------------------------------------------------
SELECT 
    p.category,
    COUNT(DISTINCT o.order_id) AS order_count,
    SUM(o.quantity) AS total_quantity_sold,
    ROUND(SUM(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS net_revenue,
    ROUND(AVG(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS avg_order_value,
    ROUND(100.0 * SUM(CASE WHEN o.returned = 1 THEN 1 ELSE 0 END) / COUNT(o.order_id), 2) AS return_rate_pct
FROM products p
INNER JOIN orders o ON p.product_id = o.product_id
GROUP BY p.category
ORDER BY net_revenue DESC;


-- --------------------------------------------------------------------
-- QUERY 4: Monthly Net Revenue Trend
-- Tracks revenue and volume trajectory month-by-month
-- --------------------------------------------------------------------
SELECT 
    STRFTIME('%Y-%m', o.order_date) AS order_month,
    COUNT(o.order_id) AS order_count,
    SUM(o.quantity) AS units_sold,
    ROUND(SUM(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS monthly_net_revenue
FROM orders o
WHERE o.order_date IS NOT NULL
GROUP BY order_month
ORDER BY order_month ASC;


-- --------------------------------------------------------------------
-- QUERY 5: Top 5 Products by Net Revenue
-- Identifies hero SKU items driving maximum top-line revenue
-- --------------------------------------------------------------------
SELECT 
    p.product_id,
    p.name AS product_name,
    p.category,
    p.brand,
    SUM(o.quantity) AS units_sold,
    ROUND(SUM(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS net_revenue
FROM products p
INNER JOIN orders o ON p.product_id = o.product_id
GROUP BY p.product_id, p.name, p.category, p.brand
ORDER BY net_revenue DESC
LIMIT 5;
