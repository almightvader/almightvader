import numpy as np, pandas as pd, pickle, json
pd.set_option("display.width",230); pd.set_option("display.max_columns",40)
raw=pd.read_csv("/mnt/user-data/outputs/glowup_campaigns.csv"); log={"raw_rows":len(raw)}
df=raw.copy()
log["duplicates"]=int(df.duplicated().sum()); df=df.drop_duplicates().reset_index(drop=True)
cm={"social media":"Social Media","social":"Social Media","influencer":"Influencer","influencers":"Influencer","email":"Email","e-mail":"Email","events":"Events","event":"Events"}
log["channel_labels_fixed"]=int((df.channel!=df.channel.str.strip().str.lower().map(cm)).sum()); df["channel"]=df.channel.str.strip().str.lower().map(cm)
# conversions > leads -> use revenue / platform median revenue per conversion (from valid rows)
ok=df[df.conversions<=df.leads]; rpc=(ok.revenue/ok.conversions.replace(0,np.nan)).groupby(ok.platform).median()
bad=df.conversions>df.leads; log["conversions_gt_leads_fixed"]=int(bad.sum())
df.loc[bad,"conversions"]=(df.loc[bad,"revenue"]/df.loc[bad,"platform"].map(rpc)).round()
# impute clicks / impressions
df["imputed"]=""
ctr=(df.clicks/df.impressions).groupby(df.platform).median(); ipr=(df.impressions/df.spend).groupby(df.platform).median()
log["missing_clicks"]=int(df.clicks.isna().sum()); log["missing_impr"]=int(df.impressions.isna().sum()); log["missing_both"]=int((df.clicks.isna()&df.impressions.isna()).sum())
a=df.clicks.isna()&df.impressions.notna(); df.loc[a,"clicks"]=(df.loc[a,"impressions"]*df.loc[a,"platform"].map(ctr)).round()
b=df.impressions.isna()&df.clicks.notna(); df.loc[b,"impressions"]=(df.loc[b,"clicks"]/df.loc[b,"platform"].map(ctr)).round()
c=df.impressions.isna()&df.clicks.isna(); df.loc[c,"impressions"]=(df.loc[c,"spend"]*df.loc[c,"platform"].map(ipr)).round(); df.loc[c,"clicks"]=(df.loc[c,"impressions"]*df.loc[c,"platform"].map(ctr)).round()
df["clicks"]=df.clicks.astype(int); df["impressions"]=df.impressions.astype(int)
df["roi"]=(df.revenue-df.spend)/df.spend*100; df["cpa"]=df.spend/df.conversions.replace(0,np.nan); df["cpl"]=df.spend/df.leads
df["lead_to_conv"]=df.conversions/df.leads; df["ctr"]=df.clicks/df.impressions; df["cpc"]=df.spend/df.clicks; df["eng_rate"]=(df.likes+df.shares)/df.impressions
df["month_dt"]=pd.to_datetime(df.month+"-01")
df.to_csv("glowup_campaigns_clean.csv",index=False)
def agg(g):
    a=g.agg(campaigns=("spend","size"),spend=("spend","sum"),impressions=("impressions","sum"),clicks=("clicks","sum"),leads=("leads","sum"),conversions=("conversions","sum"),revenue=("revenue","sum"))
    a["roi"]=(a.revenue-a.spend)/a.spend*100; a["cpa"]=a.spend/a.conversions; a["cpl"]=a.spend/a.leads; a["lead_to_conv"]=a.conversions/a.leads; a["ctr"]=a.clicks/a.impressions
    a["spend_share"]=a.spend/a.spend.sum(); a["rev_share"]=a.revenue/a.revenue.sum(); a["conv_share"]=a.conversions/a.conversions.sum(); return a
T=dict(spend=df.spend.sum(),rev=df.revenue.sum(),leads=int(df.leads.sum()),conv=int(df.conversions.sum()),imp=int(df.impressions.sum()),clicks=int(df.clicks.sum()))
T["roi"]=(T["rev"]-T["spend"])/T["spend"]*100; T["cpa"]=T["spend"]/T["conv"]; T["l2c"]=T["conv"]/T["leads"]; T["roas"]=T["rev"]/T["spend"]
ch=agg(df.groupby("channel")).sort_values("roi",ascending=False); pl=agg(df.groupby(["channel","platform"])).reset_index().sort_values("roi",ascending=False)
mo=agg(df.groupby("month")); aud=agg(df.groupby("target_audience")).sort_values("roi",ascending=False); off=agg(df.groupby("offer_type")).sort_values("roi",ascending=False)
cre=agg(df.groupby(["platform","creative_type"])).reset_index(); prod=agg(df.groupby("product_focus")).sort_values("roi",ascending=False)
# high leads low conversion
hi=df.leads.quantile(.75); lo=df.lead_to_conv.quantile(.25)
hl=df[(df.leads>=hi)&(df.lead_to_conv<=lo)].sort_values("leads",ascending=False)
print(json.dumps(log,indent=1)); print({k:round(float(v),3) for k,v in T.items()}); print("hi leads thr",hi,"lo l2c thr",lo)
print(ch.round(3).to_string()); print(pl.round(3).to_string()); print(mo.round(3)[["spend","revenue","conversions","roi","cpa"]]); 
print(aud.round(3)[["campaigns","spend","roi","cpa","lead_to_conv"]]); print(off.round(3)[["campaigns","spend","roi","cpa","lead_to_conv"]]); print(prod.round(3)[["campaigns","spend","roi"]])
print(cre.round(3)[["platform","creative_type","campaigns","roi","cpa","ctr","lead_to_conv"]])
print(hl[["campaign_id","campaign_name","leads","conversions","lead_to_conv","spend","roi","target_audience","offer_type","creative_type"]].round(3).to_string())
top=df.sort_values("roi",ascending=False); print(top[["campaign_id","campaign_name","month","spend","revenue","roi","cpa","conversions"]].head(10).round(1).to_string()); print(top[["campaign_id","campaign_name","month","spend","revenue","roi","cpa","conversions"]].tail(10).round(1).to_string())
print("neg roi campaigns",(df.roi<0).sum(),"spend on them",df[df.roi<0].spend.sum(),"share",df[df.roi<0].spend.sum()/T["spend"])
print("eng rate by platform"); print(df.groupby("platform").eng_rate.mean().round(4))
print("corr eng_rate vs roi", df.dropna(subset=["eng_rate"])[["eng_rate","roi"]].corr().iloc[0,1], " ctr vs roi", df[["ctr","roi"]].corr().iloc[0,1])
pickle.dump(dict(df=df,T=T,ch=ch,pl=pl,mo=mo,aud=aud,off=off,cre=cre,prod=prod,hl=hl,log=log),open("res.pkl","wb"))
