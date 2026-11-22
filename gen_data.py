import numpy as np, pandas as pd
rng=np.random.default_rng(11)
months=pd.period_range("2025-11","2026-10",freq="M")
season={11:1.25,12:1.25,1:.95,2:1.15,3:1.0,4:1.1,5:1.1,6:.85,7:.85,8:.85,9:.95,10:1.2}
prod_by_month={11:"Gift Sets",12:"Gift Sets",1:"Hydrating Moisturiser",2:"Lip Care & Lipstick",3:"Face Wash",4:"Sunscreen SPF50",5:"Sunscreen SPF50",6:"Sunscreen SPF50",7:"Vitamin C Serum",8:"Sheet Masks",9:"Vitamin C Serum",10:"Festive Glow Kit"}
# channel, platform, n, spend(lo,hi), cpm, ctr, click->lead, lead->conv, aov, creative options
SPEC=[
 ("Social Media","Instagram Ads",14,(60e3,180e3),120,.016,.15,.12,850,["Reel","Static","Carousel"]),
 ("Social Media","Facebook Ads",8,(50e3,140e3),90,.009,.10,.07,800,["Static","Video","Carousel"]),
 ("Social Media","YouTube Ads",5,(80e3,200e3),130,.009,.10,.12,900,["Video"]),
 ("Influencer","Nano Influencer",4,(20e3,40e3),250,.030,.22,.15,850,["Reel","Story"]),
 ("Influencer","Micro Influencer",5,(50e3,120e3),210,.022,.22,.12,850,["Reel","Story"]),
 ("Influencer","Macro Influencer",4,(150e3,300e3),140,.014,.20,.065,850,["Reel","Video"]),
 ("Influencer","Mega Influencer",3,(300e3,600e3),90,.008,.22,.025,800,["Reel","Video"]),
 ("Email","Email Newsletter",10,(8e3,25e3),0,.032,.20,.22,950,["Newsletter"]),
 ("Events","Local Event / Pop-up",7,(150e3,350e3),0,.14,.45,.35,1100,["Stall + Sampling"]),
]
aud_f={"18-24":(1.0,.8,.95),"25-34":(1.0,1.15,1.0),"35-44":(.85,1.0,1.1)}  # ctr, conv, aov
off_f={"Discount":(1.15,.9),"Free Sample":(1.25,1.0),"Bundle":(1.1,1.15),"No Offer":(.85,1.0)}
rows=[]; cid=1
for ch,pl,n,(slo,shi),cpm,ctr,lr,cr,aov,creatives in SPEC:
    for k in range(n):
        m=months[rng.integers(0,12)] if ch!="Events" else months[rng.choice([0,1,3,5,10,11,6])]
        mo=m.month; spend=float(rng.uniform(slo,shi))
        aud=rng.choice(list(aud_f),p=[.35,.45,.20] if ch!="Email" else [.15,.55,.30])
        if "Mega" in pl: aud=rng.choice(["18-24","25-34"],p=[.75,.25])
        offp={"Social Media":[.35,.05,.2,.4],"Influencer":[.25,.1,.15,.5],"Email":[.3,.2,.3,.2],"Events":[.0,.8,.2,.0]}[ch]
        off=rng.choice(["Discount","Free Sample","Bundle","No Offer"],p=offp)
        creative=rng.choice(creatives)
        fb_decay = (1-0.25*((m.year-2025)*12+mo-11)/11) if pl=="Facebook Ads" else 1
        cf=1.0
        if creative=="Static": cf=.85
        if creative in("Reel",): cf=1.15
        c_ctr=ctr*aud_f[aud][0]*off_f[off][0]**0.3*cf*season[mo]**0.4*rng.lognormal(0,.12)
        c_cr=cr*aud_f[aud][1]*off_f[off][0]*season[mo]*fb_decay*rng.lognormal(0,.18)
        c_lr=lr*rng.lognormal(0,.1)
        c_aov=aov*aud_f[aud][2]*off_f[off][1]*rng.lognormal(0,.07)
        if ch in("Email",): imp=int(spend/ (0.00045) * 0.0+ rng.integers(20000,60000))
        elif ch=="Events": imp=int(rng.integers(4000,12000)*(spend/250e3)**0.5)
        else: imp=int(spend/cpm*1000*rng.lognormal(0,.1))
        clicks=int(imp*c_ctr); leads=int(clicks*c_lr); conv=int(round(leads*c_cr)); rev=round(conv*c_aov,0)
        if ch in("Social Media","Influencer"):
            eng=rng.uniform(.02,.06); likes=int(imp*eng); shares=int(likes*rng.uniform(.04,.09))
        else: likes=shares=np.nan
        dur=int(rng.integers(2,4)) if ch=="Events" else int(rng.integers(7,29))
        sd=pd.Timestamp(m.year,mo,int(rng.integers(1,max(2,29-dur))))
        name={"Gift Sets":"Festive Gifting","Hydrating Moisturiser":"Winter Hydration","Lip Care & Lipstick":"Love Your Lips","Face Wash":"Fresh Start","Sunscreen SPF50":"Sun Shield","Vitamin C Serum":"Vitamin C Glow","Sheet Masks":"Mask Monday","Festive Glow Kit":"Festive Glow"}[prod_by_month[mo]]
        rows.append([f"CMP{cid:03d}",f"{name} - {pl}",ch,pl,sd.date().isoformat(),(sd+pd.Timedelta(days=dur)).date().isoformat(),sd.strftime("%Y-%m"),prod_by_month[mo],aud,creative,off,round(spend),imp,clicks,leads,conv,rev,likes,shares]); cid+=1
df=pd.DataFrame(rows,columns=["campaign_id","campaign_name","channel","platform","start_date","end_date","month","product_focus","target_audience","creative_type","offer_type","spend","impressions","clicks","leads","conversions","revenue","likes","shares"])
df=df.sample(frac=1,random_state=3).sort_values("start_date").reset_index(drop=True); df["campaign_id"]=[f"CMP{i+1:03d}" for i in range(len(df))]
n=len(df)
# mess
cv={"Social Media":["social media","Social media ","SOCIAL MEDIA","Social"],"Influencer":["influencer","Influencers","INFLUENCER "],"Email":["email","E-mail","EMAIL "],"Events":["events","Event","EVENTS"]}
mk=rng.random(n)<.10; df.loc[mk,"channel"]=[rng.choice(cv[c]) for c in df.loc[mk,"channel"]]
ci=rng.choice(n,6,replace=False); df.loc[ci,"clicks"]=np.nan
ii=rng.choice(n,5,replace=False); df.loc[ii,"impressions"]=np.nan
both=rng.choice(n,2,replace=False); df.loc[both,["clicks","impressions"]]=np.nan
small=df.index[df.leads<500].to_numpy(); bad=rng.choice(small,2,replace=False); df.loc[bad,"conversions"]=df.loc[bad,"leads"]+rng.integers(1,6,2)   # conversions > leads (entry error)
df=pd.concat([df,df.sample(2,random_state=5)]).sort_values("start_date").reset_index(drop=True)
df.to_csv("/mnt/user-data/outputs/glowup_campaigns.csv",index=False)
print(df.shape); print(df.isna().sum()); print(df.channel.value_counts().head(20))
g=df.assign(ch=df.platform).groupby("ch").agg(n=("spend","size"),spend=("spend","sum"),rev=("revenue","sum"),conv=("conversions","sum"),leads=("leads","sum")); g["roi"]=(g.rev-g.spend)/g.spend*100; g["cpa"]=g.spend/g.conv; print(g.round(0)); print("total spend",df.spend.sum(),"rev",df.revenue.sum())
