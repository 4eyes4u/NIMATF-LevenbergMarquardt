# NIMATF-LevenbergMarquardt

U okviru projekta je ručno implementiran Levenberg-Marquardt algoritam za rešavanje problema najmanjih kvadrata
nelinearnog tipa. Za demonstraciju rada algoritma korišćene su dve funkcije - jedna za koju se očekuje gotovo potpuno
precizna aproksimacija i druga koja je u tom smislu daleko komplikovanija. Dobijeni rezultati su upoređeni sa onima
koje vraća srodna funkcija iz `scipy` modula.

Projekat je urađen u sklopu kursa Naučno izračunavanje na Matematičkom fakultetu, Univerziteta u Beogradu. U izradi
je učestvovao Kosta Grujčić (1012/2021).

## Sadržaj
Projekat se sastoji iz jedne sveske u kojoj se nalaze implementacija algoritma i prateća teorija. Osnovna primena je
aproksimacija nelinearnih funkcija, a korišćene su:
* $\exp(ax + b)$ - označena kao lakša i
* $a\sin(bx) + b\cos(ax)$ - označena kao teža.

## Pokretanje
Projekat je implementiran kao jupyter sveska, pa je preporučeno sprovesti sledeće korake u korenu repozitorijuma:

1. `$ conda env create --file environment.yml`
2. `$ conda activate nimatf-ml`
3. `$ jupyter notebook`