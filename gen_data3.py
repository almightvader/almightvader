import numpy as np, pandas as pd
rng = np.random.default_rng(42)

# name, category, price, cost, popularity, true elasticity, mu0, promo_prob, discount level weights (5,10,15,20,25)
P = [
 ("Classic Veg Burger","Burgers",129,62,9,-1.3,1.5,.25,[.3,.4,.2,.1,0]),
 ("Chicken Zinger Burger","Burgers",189,105,8,-1.0,1.5,.25,[.3,.4,.2,.1,0]),
 ("Double Cheese Burger","Burgers",219,128,5,-1.6,1.4,.30,[.2,.3,.3,.2,0]),
 ("Margherita Pizza","Pizza",249,118,8,-1.2,1.5,.30,[.2,.3,.3,.2,0]),
 ("Farmhouse Pizza","Pizza",349,175,5,-1.3,1.4,.30,[.2,.3,.3,.2,0]),
 ("Cheese Burst Pizza","Pizza",399,262,7,-1.7,1.5,.55,[0,.1,.2,.3,.4]),
 ("Chicken Tikka Pizza","Pizza",379,190,5,-1.2,1.4,.30,[.2,.3,.3,.2,0]),
 ("Paneer Butter Masala","Indian Mains",279,130,6,-0.9,1.4,.15,[.4,.4,.2,0,0]),
 ("Chicken Biryani","Indian Mains",299,172,9,-0.6,1.6,.12,[.5,.4,.1,0,0]),
 ("Veg Thali","Indian Mains",199,121,6,-1.1,1.5,.30,[.2,.3,.3,.2,0]),
 ("Dal Makhani","Indian Mains",229,95,4,-0.8,1.4,.15,[.4,.4,.2,0,0]),
 ("French Fries","Sides",99,32,10,-1.4,1.6,.35,[.2,.3,.3,.2,0]),
 ("Garlic Bread","Sides",119,42,5,-1.0,1.4,.25,[.3,.4,.3,0,0]),
 ("Chicken Wings","Sides",229,142,6,-1.5,1.4,.40,[.1,.2,.3,.3,.1]),
 ("Cola 500ml","Beverages",60,46,11,-1.8,1.6,.50,[0,.1,.2,.3,.4]),
 ("Cold Coffee","Beverages",129,38,6,-0.6,1.4,.10,[.5,.4,.1,0,0]),
 ("Fresh Lime Soda","Beverages",79,22,5,-0.7,1.4,.12,[.5,.4,.1,0,0]),
 ("Mango Shake","Beverages",149,78,4,-1.1,1.4,.25,[.3,.4,.3,0,0]),
 ("Chocolate Brownie","Desserts",129,45,5,-0.9,1.4,.20,[.4,.4,.2,0,0]),
 ("Gulab Jamun (2 pc)","Desserts",89,28,4,-0.7,1.4,.15,[.4,.4,.2,0,0]),
 ("Ice Cream Sundae","Desserts",149,108,4,-1.6,1.4,.45,[.1,.2,.3,.3,.1]),
 ("Burger Combo","Combos",299,190,5,-1.9,1.5,.55,[0,.1,.2,.3,.4]),
 ("Family Meal Combo","Combos",799,540,3,-1.5,1.3,.55,[0,.1,.2,.3,.4]),
 ("Biryani + Raita Combo","Combos",349,205,4,-1.0,1.4,.35,[.2,.3,.3,.2,0]),
]
levels = np.array([5,10,15,20,25])
cities = ["Delhi NCR","Mumbai","Bengaluru","Hyderabad","Pune","Chennai"]
cw = np.array([.26,.22,.20,.12,.10,.10])
start, end = pd.Timestamp("2024-11-01"), pd.Timestamp("2026-10-31")
days = pd.date_range(start,end)
dow_w = np.array([0.9,0.9,0.9,1.0,1.2,1.4,1.3])
day_w = np.array([dow_w[d.dayofweek]*(1+0.25*(d>=pd.Timestamp("2025-11-01"))*0+ (d-start).days/730*0.35) for d in days])
day_w /= day_w.sum()
N_ORD = 24000
order_days = rng.choice(days, N_ORD, p=day_w)
pop = np.array([p[4] for p in P],float); pop/=pop.sum()
rows=[]
for oid, od in enumerate(order_days, start=100001):
    od = pd.Timestamp(od)
    cust = f"C{rng.integers(1,7001):05d}"
    city = rng.choice(cities,p=cw)
    n_items = rng.choice([1,2,3,4],p=[.35,.35,.2,.1])
    idx = rng.choice(len(P), n_items, replace=False, p=pop)
    # campaign weeks boost promo odds
    campaign = 1.25 if (od.month in (1,5,10) and od.day<=14) else 1.0
    for i in idx:
        nm,cat,price,cost,_,e,mu0,pp,lw = P[i]
        promo = rng.random() < min(pp*campaign,.95)
        d = int(rng.choice(levels,p=lw)) if promo else 0
        mu = mu0*((1-d/100)**e)
        qty = 1 + rng.poisson(max(mu-1,0.05))
        unit_cost = cost*(1.04 if od>=pd.Timestamp("2025-11-01") else 1.0)
        rows.append([oid,od.date().isoformat(),cust,city,f"P{i+1:02d}",nm,cat,int(qty),price,round(unit_cost,2),d,"Yes" if promo else "No"])
df = pd.DataFrame(rows,columns=["order_id","order_date","customer_id","city","product_id","product_name","category","quantity","unit_price","unit_cost","discount_pct","promo_applied"])

# ---- inject real-world messiness ----
n=len(df)
variants = {"Burgers":["burger","BURGERS","Burger "],"Pizza":["pizza","PIZZA","Pizzas"],"Indian Mains":["Indian Main","indian mains","Mains"],
 "Sides":["Side","sides","SIDES "],"Beverages":["Beverage","beverages","Drinks","BEVERAGES"],"Desserts":["Dessert","desserts","Sweets"],"Combos":["Combo","combos","COMBOS"]}
m = rng.random(n)<0.07
df.loc[m,"category"] = [rng.choice(variants[c]) for c in df.loc[m,"category"]]
# a few genuinely wrong categories
w = rng.random(n)<0.004
df.loc[w,"category"] = rng.choice(["Snacks","Food","Misc"],w.sum())
df["discount_pct"] = df["discount_pct"].astype(float)
miss = rng.random(n)<0.04
df.loc[miss,"discount_pct"] = np.nan
bad = rng.choice(n,35,replace=False)
df.loc[bad,"discount_pct"] = rng.choice([30,35,40,50],35)   # policy violations / entry errors
dups = df.sample(60,random_state=1)
df = pd.concat([df,dups]).sample(frac=1,random_state=7).sort_values(["order_date","order_id"]).reset_index(drop=True)
df.to_csv("/mnt/user-data/outputs/tastybites_orders.csv",index=False)
print(df.shape); print(df.head(8)); print(df.isna().sum()); print(df.category.value_counts())
