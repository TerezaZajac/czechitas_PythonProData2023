import pandas as pd
import matplotlib.pyplot as plt

presidents = pd.read_csv('1976-2020-president.csv')

#presidents = presidents[['year', 'state', 'party_simplified', 'candidatevotes', 'totalvotes' ]]
#Urči pořadí jednotlivých kandidátů v jednotlivých státech a v jednotlivých letech (pomocí metody rank()). Nezapomeň, že data je před použitím metody nutné seřadit a spolu s metodou rank() je nutné použít metodu groupby().
presidents['rank'] = presidents.groupby(['state', 'year'])["candidatevotes"].rank(method='min', ascending=False)

#Pro další analýzu jsou důležití pouze vítězové. Vytvoř novou tabulku, která bude obsahovat pouze vítěze voleb.
presidents_winners = presidents[presidents['rank'] == 1]

#Pomocí metody shift() přidej nový sloupec, abys v jednotlivých řádcích měl(a) po sobě vítězné strany ve dvou po sobě jdoucích letech.
presidents_winners = presidents_winners.sort_values(['state', 'year'])
presidents_winners['previous_winner_party'] = presidents_winners['party_simplified'].shift(periods=1)

#Porovnej, jestli se ve dvou po sobě jdoucích letech změnila vítězná strana. Můžeš k tomu použít např. funkci numpy.where() nebo metodu apply().

def winner_change(row):
    
    if row['year'] == 1976:
        return 0
    elif row['party_simplified'] == row['previous_winner_party']:
        return 0
    else:
        return 1
    
presidents_winners['change'] = presidents_winners.apply(winner_change, axis=1)

#Proveď agregaci podle názvu státu a seřaď státy podle počtu změn vítězných stran.

presidents_winners_pivot = presidents_winners.groupby(['state'])['change'].sum()
presidents_winners_pivot = pd.DataFrame(presidents_winners_pivot)
presidents_winners_pivot = presidents_winners_pivot.sort_values('change', ascending=False)
#print(presidents_winners_pivot.head(15))

#Vytvoř sloupcový graf s 10 státy, kde došlo k nejčastější změně vítězné strany. Jako výšku sloupce nastav počet změn.
presidents_winners_plot = presidents_winners_pivot[presidents_winners_pivot['change'] >= 3]
presidents_winners_plot.plot(kind="bar")
#plt.show()


# Přidej do tabulky sloupec, který obsahuje absolutní rozdíl mezi vítězem a druhým v pořadí.
presidents_two_best = presidents[presidents['rank'] < 3]
presidents_two_best = presidents_two_best.sort_values(['year', 'state', 'rank'])
presidents_two_best['second_candidate_votes'] = presidents_two_best['candidatevotes'].shift(periods=-1)
presidents_best = presidents_two_best[presidents_two_best['rank'] == 1]
presidents_best['margin'] = presidents_best['candidatevotes'] - presidents_best['second_candidate_votes']

#print(presidents_best.tail())


# Přidej sloupec s relativním marginem, tj. rozdílem vyděleným počtem hlasů.
presidents_best['relative_margin'] = presidents_best['margin'] / presidents_best['totalvotes']
#print(presidents_best.head())

# Seřaď tabulku podle velikosti relativního marginu a zjisti, kdy a ve kterém státě byl výsledek voleb nejtěsnější.
presidents_best_sorted = presidents_best.sort_values(['relative_margin'])
#print(presidents_best_sorted.head())

# Vytvoř pivot tabulku, která zobrazí pro jednotlivé volební roky, kolik států přešlo od Republikánské strany k Demokratické straně, kolik států přešlo od Demokratické strany k Republikánské straně a kolik států volilo kandidáta stejné strany.


def winner_swing(x):
    if x['change'] == 1 and x['previous_winner_party'] == 'DEMOCRAT':
        return 'to Rep.'
    elif x['change'] == 1 and x['previous_winner_party'] == 'REPUBLICAN':
        return 'to Dem.'
    else:
        return 'no swing'
    
presidents_winners['swing'] = presidents_winners.apply(winner_swing, axis=1)
#print(presidents_winners.head())

presidents_pivot = pd.pivot_table(data = presidents_winners , values="state" , index="year" , columns="change" , aggfunc=len, fill_value=0)


print(presidents_pivot)

#omlouvam se tu pivot tabulku proste nevim :/


