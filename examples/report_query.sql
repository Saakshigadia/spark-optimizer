-- Monthly revenue report (first version)
SELECT *
FROM orders o, customers c
WHERE o.customer_id = c.id
  AND year(o.order_date) = 2026
  AND o.id NOT IN (SELECT order_id FROM refunds)
UNION
SELECT * FROM orders_archive
ORDER BY amount DESC;
