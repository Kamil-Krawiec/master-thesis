CREATE TABLE categories (
    category_id   SMALLSERIAL PRIMARY KEY,
    category_name VARCHAR(15)  NOT NULL,
    description   TEXT,
    picture       BYTEA
);

CREATE TABLE customer_demographics (
    customer_type_id CHAR(10) PRIMARY KEY,
    customer_desc    TEXT
);

CREATE TABLE customers (
    customer_id   CHAR(10)  PRIMARY KEY,
    company_name  VARCHAR(40) NOT NULL,
    contact_name  VARCHAR(30),
    contact_title VARCHAR(30),
    address       VARCHAR(60),
    city          VARCHAR(15),
    region        VARCHAR(15),
    postal_code   VARCHAR(10),
    country       VARCHAR(15),
    phone         VARCHAR(24),
    fax           VARCHAR(24)
);

CREATE TABLE customer_customer_demo (
    customer_id      CHAR(10) NOT NULL,
    customer_type_id CHAR(10) NOT NULL,
    CONSTRAINT pk_customer_customer_demo PRIMARY KEY (customer_id, customer_type_id),
    CONSTRAINT fk_cccd_customer FOREIGN KEY (customer_id)
        REFERENCES customers (customer_id),
    CONSTRAINT fk_cccd_cust_demo FOREIGN KEY (customer_type_id)
        REFERENCES customer_demographics (customer_type_id)
);

CREATE TABLE employees (
    employee_id     SMALLSERIAL PRIMARY KEY,
    last_name       VARCHAR(20) NOT NULL,
    first_name      VARCHAR(10) NOT NULL,
    title           VARCHAR(30),
    title_of_courtesy VARCHAR(25),
    birth_date      DATE,
    hire_date       DATE,
    address         VARCHAR(60),
    city            VARCHAR(15),
    region          VARCHAR(15),
    postal_code     VARCHAR(10),
    country         VARCHAR(15),
    home_phone      VARCHAR(24),
    extension       VARCHAR(4),
    photo           BYTEA,
    notes           TEXT,
    reports_to      SMALLINT,
    photo_path      VARCHAR(255)
);

CREATE TABLE suppliers (
    supplier_id   SMALLSERIAL PRIMARY KEY,
    company_name  VARCHAR(40) NOT NULL,
    contact_name  VARCHAR(30),
    contact_title VARCHAR(30),
    address       VARCHAR(60),
    city          VARCHAR(15),
    region        VARCHAR(15),
    postal_code   VARCHAR(10),
    country       VARCHAR(15),
    phone         VARCHAR(24),
    fax           VARCHAR(24),
    homepage      TEXT
);

CREATE TABLE products (
    product_id        SMALLSERIAL PRIMARY KEY,
    product_name      VARCHAR(40) NOT NULL,
    supplier_id       SMALLINT,
    category_id       SMALLINT,
    quantity_per_unit VARCHAR(20),
    unit_price        NUMERIC(10,2),
    units_in_stock    SMALLINT,
    units_on_order    SMALLINT,
    reorder_level     SMALLINT,
    discontinued      BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT fk_products_supplier FOREIGN KEY (supplier_id)
        REFERENCES suppliers (supplier_id),
    CONSTRAINT fk_products_category FOREIGN KEY (category_id)
        REFERENCES categories (category_id)
);

CREATE TABLE region (
    region_id          SMALLSERIAL PRIMARY KEY,
    region_description VARCHAR(50) NOT NULL
);

CREATE TABLE shippers (
    shipper_id  SMALLSERIAL PRIMARY KEY,
    company_name VARCHAR(40) NOT NULL,
    phone        VARCHAR(24)
);

CREATE TABLE orders (
    order_id      SERIAL PRIMARY KEY,
    customer_id   CHAR(10),
    employee_id   SMALLINT,
    order_date    DATE,
    required_date DATE,
    shipped_date  DATE,
    ship_via      SMALLINT,
    freight       NUMERIC(10,2),
    ship_name     VARCHAR(40),
    ship_address  VARCHAR(60),
    ship_city     VARCHAR(15),
    ship_region   VARCHAR(15),
    ship_postal_code VARCHAR(10),
    ship_country  VARCHAR(15),
    CONSTRAINT fk_orders_customer FOREIGN KEY (customer_id)
        REFERENCES customers (customer_id),
    CONSTRAINT fk_orders_employee FOREIGN KEY (employee_id)
        REFERENCES employees (employee_id),
    CONSTRAINT fk_orders_shipper FOREIGN KEY (ship_via)
        REFERENCES shippers (shipper_id)
);

CREATE TABLE territories (
    territory_id          VARCHAR(20) PRIMARY KEY,
    territory_description VARCHAR(50) NOT NULL,
    region_id             SMALLINT,
    CONSTRAINT fk_territories_region FOREIGN KEY (region_id)
        REFERENCES region (region_id)
);

CREATE TABLE employee_territories (
    employee_id  SMALLINT,
    territory_id VARCHAR(20),
    CONSTRAINT pk_employee_territories PRIMARY KEY (employee_id, territory_id),
    CONSTRAINT fk_et_employee FOREIGN KEY (employee_id)
        REFERENCES employees (employee_id),
    CONSTRAINT fk_et_territory FOREIGN KEY (territory_id)
        REFERENCES territories (territory_id)
);

CREATE TABLE order_details (
    order_id   INTEGER,
    product_id SMALLINT,
    unit_price NUMERIC(10,2) NOT NULL,
    quantity   SMALLINT      NOT NULL,
    discount   NUMERIC(4,2)   NOT NULL DEFAULT 0,
    CONSTRAINT pk_order_details PRIMARY KEY (order_id, product_id),
    CONSTRAINT fk_order_details_order FOREIGN KEY (order_id)
        REFERENCES orders (order_id),
    CONSTRAINT fk_order_details_product FOREIGN KEY (product_id)
        REFERENCES products (product_id)
);

CREATE TABLE us_states (
    state_id     SMALLSERIAL PRIMARY KEY,
    state_name   VARCHAR(100),
    state_abbr   CHAR(2),
    state_region VARCHAR(50)
);