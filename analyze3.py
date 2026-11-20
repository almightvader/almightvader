import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
pd.set_option("display.width",220); pd.set_option("display.max_columns",30)
raw = pd.read_csv("/mnt/user-data/outputs/tastybites_orders.csv")
log = {"raw_rows":len(raw)}

# ---------- CLEANING ----------
df = raw.copy()
log["duplicates_removed"] = int(df.duplicated().sum()); df = df.drop_duplicates().copy()
cmap = {"burger":"Burgers","burgers":"Burgers","pizza":"Pizza","pizzas":"Pizza","indian main":"Indian Mains","indian mains":"Indian Mains","mains":"Indian Mains",
        "side":"Sides","sides":"Sides","beverage":"Beverages","beverages":"Beverages","drinks":"Beverages","dessert":"Desserts","desserts":"Desserts","sweets":"Desserts",
        "combo":"Combos","combos":"Combos"}
df["cat_raw"]=df["category"]; df["category"]=df["category"].str.strip().str.lower().map(cmap)
master = df.dropna(subset=["category"]).groupby("product_id")["category"].agg(lambda s:s.mode()[0])
log["category_unmapped_fixed_via_product_master"] = int(df["category"].isna().sum())
df["category"]=df["category"].fillna(df["product_id"].map(master))
log["category_label_variants_standardised"] = int((df["cat_raw"].str.strip()!=df["category"]).sum())
log["missing_discount"] = int(df["discount_pct"].isna().sum())
promo_med = df[(df.promo_applied=="Yes")&df.discount_pct.notna()].groupby("product_id")["discount_pct"].median()
m = df.discount_pct.isna()
df.loc[m&(df.promo_applied=="No"),"discount_pct"]=0
mm = m&(df.promo_applied=="Yes"); df.loc[mm,"discount_pct"]=df.loc[mm,"product_id"].map(promo_med)
log["discount_over_25_capped"] = int((df.discount_pct>25).sum()); df["discount_pct"]=df["discount_pct"].clip(upper=25)
log["clean_rows"]=len(df)
df["order_date"]=pd.to_datetime(df["order_date"]); df["month"]=df.order_date.dt.to_period("M").dt.to_timestamp()
df["gross"]=df.quantity*df.unit_price; df["net"]=df.gross*(1-df.discount_pct/100); df["cost"]=df.quantity*df.unit_cost
df["profit"]=df.net-df.cost; df["disc_amt"]=df.gross-df.net
df.to_csv("tastybites_orders_clean.csv",index=False)

T = dict(rev=df.net.sum(),profit=df.profit.sum(),gross=df.gross.sum(),disc=df.disc_amt.sum(),units=int(df.quantity.sum()),orders=df.order_id.nunique())
T["margin"]=T["profit"]/T["rev"]; T["aov"]=df.groupby("order_id").net.sum().mean(); T["disc_rate"]=T["disc"]/T["gross"]
T["promo_share_lines"]=(df.discount_pct>0).mean()

def agg(g):
    a=g.agg(units=("quantity","sum"),gross=("gross","sum"),revenue=("net","sum"),cost=("cost","sum"),profit=("profit","sum"),disc=("disc_amt","sum"),lines=("quantity","size"))
    a["margin"]=a.profit/a.revenue; a["profit_share"]=a.profit/a.profit.sum(); a["rev_share"]=a.revenue/a.revenue.sum()
    a["avg_disc_rate"]=a.disc/a.gross; return a
prod = agg(df.groupby(["product_name","category"])).reset_index().sort_values("profit",ascending=False)
prod["profit_rank"]=prod.profit.rank(ascending=False).astype(int); prod["units_rank"]=prod.units.rank(ascending=False).astype(int)
cat = agg(df.groupby("category")).reset_index().sort_values("profit",ascending=False)
prod["cum_profit"]=prod.profit_share.cumsum()

# ---------- discount level / lift ----------
df["d"]=df.discount_pct
lvl = df.groupby("d").agg(lines=("quantity","size"),avg_qty=("quantity","mean"),margin=("profit","sum"),rev=("net","sum"),profit=("profit","sum"))
lvl["margin"]=lvl.profit/lvl.rev
# per product baseline qty & incremental promo profit
q0 = df[df.d==0].groupby("product_id").quantity.mean()
df["q0"]=df.product_id.map(q0)
df["cf_profit"]=df.q0*(df.unit_price-df.unit_cost)       # profit if the line had been sold at full price at baseline volume
pr = df[df.d>0].copy(); pr["incr"]=pr.profit-pr.cf_profit
promo_prod = pr.groupby(["product_name","category"]).agg(promo_lines=("incr","size"),discount_given=("disc_amt","sum"),incr_profit=("incr","sum"),avg_disc=("d","mean")).reset_index().sort_values("incr_profit")
promo_cat = pr.groupby("category").agg(discount_given=("disc_amt","sum"),incr_profit=("incr","sum"),avg_disc=("d","mean")).reset_index().sort_values("incr_profit")
# margin by discount level per category
cat_lvl = df.assign(band=pd.cut(df.d,[-1,0,10,15,20,25],labels=["0%","5-10%","11-15%","16-20%","21-25%"])).groupby(["category","band"],observed=True).agg(profit=("profit","sum"),rev=("net","sum")).reset_index()
cat_lvl["margin"]=cat_lvl.profit/cat_lvl.rev

# ---------- elasticity per product ----------
rows=[]
for (pid,name),g in df.groupby(["product_id","product_name"]):
    s=g.groupby("d").quantity.agg(["mean","size"]); 
    if 0 not in s.index: continue
    s=s[s["size"]>=40]; base=s.loc[0,"mean"]; s=s[s.index>0]
    x=np.log(1-s.index/100); y=np.log(s["mean"]/base); w=s["size"]
    e = (w*x*y).sum()/(w*x*x).sum() if len(s)>0 else np.nan
    rows.append((name,e,int(g[g.d>0].shape[0])))
el = pd.DataFrame(rows,columns=["product_name","elasticity","promo_lines"])
prod = prod.merge(el,on="product_name",how="left")
unit_price = df.groupby("product_name").unit_price.first(); prod["price"]=prod.product_name.map(unit_price)
prod["avg_cost"]=prod.cost/prod.units
# price scenario
def scen(r,x):
    e=r.elasticity; p0=r.price; c=r.avg_cost; dq=(1+x)**e
    base=(p0-c); new=(p0*(1+x)-c)*dq
    return (new/base-1, dq-1)
for x in (0.05,0.10):
    t=prod.apply(lambda r:scen(r,x),axis=1)
    prod[f"dprofit_{int(x*100)}"]=[a for a,b in t]; prod[f"dunits_{int(x*100)}"]=[b for a,b in t]

# ---------- AOV / trend / city ----------
oa = df.groupby("order_id").agg(net=("net","sum"),items=("quantity","sum"),disc=("disc_amt","sum"),gross=("gross","sum"),city=("city","first"),month=("month","first"),anyd=("d",lambda s:(s>0).any()))
aov_city = oa.groupby("city").net.mean().sort_values(ascending=False)
aov_disc = oa.groupby("anyd").net.mean()
mon = df.groupby("month").agg(rev=("net","sum"),profit=("profit","sum"),gross=("gross","sum"),disc=("disc_amt","sum"))
mon["margin"]=mon.profit/mon.rev; mon["disc_rate"]=mon.disc/mon.gross
yr = df.assign(y=np.where(df.order_date>="2025-11-01","Year 2 (Nov25-Oct26)","Year 1 (Nov24-Oct25)")).groupby("y").agg(rev=("net","sum"),profit=("profit","sum"),gross=("gross","sum"),disc=("disc_amt","sum"))
yr["margin"]=yr.profit/yr.rev; yr["disc_rate"]=yr.disc/yr.gross

# ---------- pairs ----------
ords = df.groupby("order_id").product_name.apply(lambda s:sorted(set(s)))
from itertools import combinations
from collections import Counter
cnt=Counter(); 
for L in ords:
    for a,b in combinations(L,2): cnt[(a,b)]+=1
pairs = pd.DataFrame([(a,b,n) for (a,b),n in cnt.items()],columns=["a","b","orders"]).sort_values("orders",ascending=False)

# ---------- print ----------
print(json.dumps(log,indent=1)); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in T.items()})
print(prod[["product_name","category","units","revenue","profit","margin","profit_share","avg_disc_rate","elasticity","units_rank","profit_rank","dprofit_5","dunits_5","dprofit_10","dunits_10"]].round(3).to_string())
print(cat.round(3).to_string()); print(lvl.round(3)); print(promo_prod.round(0).to_string()); print(promo_cat.round(0))
print(cat_lvl.pivot(index="category",columns="band",values="margin").round(3))
print(aov_city.round(1)); print(aov_disc.round(1)); print(yr.round(3)); print(mon[["margin","disc_rate"]].iloc[[0,6,12,18,23]].round(3))
print(pairs.head(8))
import pickle; pickle.dump(dict(T=T,log=log,prod=prod,cat=cat,lvl=lvl,promo_prod=promo_prod,promo_cat=promo_cat,cat_lvl=cat_lvl,aov_city=aov_city,aov_disc=aov_disc,mon=mon,yr=yr,pairs=pairs,oa=oa),open("res.pkl","wb"))
