
import pandas as pd
from scipy import stats


# Inflace
countries = pd.read_csv('python2/czechitas_PythonProData2023/countries.csv')
inflace = pd.read_csv('python2/czechitas_PythonProData2023/ukol_02_a.csv')
inflace_EU = pd.merge(inflace, countries , on=['Country'], how='inner')
#print(inflace_EU)

# Test normality
    # H0: Soubor má normální rozdělení (nulová hypotéza)
    # H1: Soubor nemá normální rozdělení (alternativní hypotéza)
res1 = stats.shapiro(inflace_EU['98'])
res2 = stats.shapiro(inflace_EU['97'])
res1 
#ShapiroResult(statistic=0.9399222135543823, pvalue=0.12131019681692123) 
    #p>0,05 H0 nezamitame
res2 
#ShapiroResult(statistic=0.952153205871582, pvalue=0.24169494211673737)
    #p>0,05 H0 nezamitame

# Formulace hypotéz testu.
    # H0: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy, se nezměnilo.
    # H1: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy, se změnilo.

# Výběr vhodného testu: porovnani te same skupinu států, dve různa časova období, normalni rozdeleni: 
    # paired T test
inflace_EU_res = stats.ttest_rel(inflace_EU['97'], inflace_EU['98'])
inflace_EU_res 
# Formulace výsledku testu (na základě p-hodnoty).
    # TtestResult(statistic=3.4869444202944764, pvalue=0.0017533857526091583, df=26)
    # p-hodnota < 0.05,  zamítáme H0, plati tedy, ze procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy, se změnilo.




#Důvěra ve stát a v EU
duvera = pd.read_csv('python2/czechitas_PythonProData2023/ukol_02_b.csv')
duvera_EU = pd.merge(duvera, countries , on=['Country'], how='inner')
#print(duvera_EU)

#Test normality
    # H0: Data mají normální rozdělení
    # H1: Data nemají normální rozdělení

res3 = stats.shapiro(duvera_EU['National Government Trust'])
res4 = stats.shapiro(duvera_EU['EU Trust'])
res3 
#ShapiroResult(statistic=0.9438267350196838, pvalue=0.15140558779239655) 
    #p>0,05 H0 nezamitame
res4 
#ShapiroResult(statistic=0.9735807180404663, pvalue=0.6981646418571472)
    #p>0,05 H0 nezamitame

# Formulace hypotéz testu.
    # H0: Důvěra lidi v národní vládu a důvěra v EU nejsou statisticky závislé.
    # H1: Důvěra lidi v národní vládu a důvěra v EU jsou statisticky závislé."

# Výběr vhodného testu: normalni rozdeleni, korelace: 
    # test zalozeny na Pearsonove korelacnim koeficientu.

duvera_EU_res = stats.pearsonr(duvera_EU['National Government Trust'], duvera_EU['EU Trust'])
duvera_EU_res
# Formulace výsledku testu (na základě p-hodnoty).
    # PearsonRResult(statistic=0.6097186340024556, pvalue=0.0007345896228823398)
    #p-hodnota < 0.05,  zamítáme H0, plati tedy, ze duvera lidi v narodni vladu a duvera lidi v EU jsou statisticke zavisle.




# Důvěra v EU a euro
#Test normality se ridi vysledkem z predesle casti. 

# zeme v Eurozone
staty_euroz = duvera_EU[duvera_EU["Euro"] == 1]
# země mimo Eurozónu
staty_mimo = duvera_EU[duvera_EU["Euro"] == 0]

# Formulace hypotéz testu.
    # H0: Důvěra v EU ve státech v eurozóně a ve státech mimo eurozónu se neliší.
    # H1: Důvěra v EU ve státech v eurozóně a ve státech mimo eurozónu se liší. 

# Výběr vhodného testu: dva soubory, neparova data, prumer:
    # Two Group T Test (neparovy T Test)

duvera_EU2_res = stats.ttest_ind(staty_euroz["EU Trust"], staty_mimo["EU Trust"])
print(duvera_EU2_res)
# Formulace výsledku testu (na základě p-hodnoty).
    # Ttest_indResult(statistic=-0.33471431258258433, pvalue=0.740632683274883)
    #H0 nezamitame, takze jsme nepotvrdili, že by se důvěra lišila.














