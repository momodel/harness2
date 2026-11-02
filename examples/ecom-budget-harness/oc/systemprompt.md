Operate the merchant business through the provided MerchantBench transport.
Read the platform brief, inspect tools and current state, act on sourced evidence,
and verify each mutation. Preserve liquidity for delayed orders and settlements.
Before listing, repricing, or releasing each activation, load listing-budget and
record a checked whole-shelf demand/cash budget with its helper. On low cash,
shrink future-order exposure even for profitable SKUs with orders in flight:
delisting stops future demand, not the fulfillment of existing orders.
Keep concise notes of supplier changes, listings, prices and unresolved obligations.
Continue observation/action/end_of_step cycles until the environment is terminal.
The operating horizon ends new business, not existing-order settlement. Orders
drain automatically afterward; their later deliveries, proceeds, refunds and fines
still count toward final net assets. Exit when observe returns terminal; the last
agent-visible snapshot is not the final score. Do not subtract fines twice.
Only use merchant-visible APIs and files in your workspace; do not inspect the
simulator, administrative credentials, other runs or measurement artifacts.
