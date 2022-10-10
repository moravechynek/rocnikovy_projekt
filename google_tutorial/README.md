# Google Developers: Text Classification

https://developers.google.com/machine-learning/guides/text-classification/

### Požadavky:
Stáhnout data z http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz (80.2MB komprimované, 210MB nekomprimované)

## Statistika dat

| Metrika                      | Hodnota       |
| ---------------------------- | ------------- |
| Počet recenzí                | 25000         |
| Počet tříd                   | 2             |
| Počet recenzí ve třídě       | 12500         |
| Medián počtu slov v recenzi  | 173           |

## Grafy

![This is an image](/google_tutorial/imgs/Figure_1.png)
![This is an image](/google_tutorial/imgs/Figure_2.png)

## Chyby

Vektory mají nesprávný datový typ

`TypeError: Dimension value must be integer or None or have an __index__ method, got value '<25000x20000 sparse matrix of type '<class 'numpy.int32'>'
        with 3482050 stored elements in Compressed Sparse Row format>' with type '<class 'scipy.sparse._csr.csr_matrix'>'`

<details><summary>OS Windows nedekóduje tyto soubory (vyřešeno):</summary>

| /train/pos/ | /train/neg | /test/pos/  | /test/neg   |
| ----------- | ---------- | ----------- |------------ |
| 10327_7.txt | 3832_4.txt | 10267_7.txt | 11356_3.txt |
| 11351_9.txt | 4526_4.txt | 10923_7.txt | 3696_4.txt  |
| 11668_7.txt | 6929_1.txt | 11046_9.txt | 688_4.txt   |
| 2362_9.txt  |            | 3554_10.txt | 6970_1.txt  |
| 4972_9.txt  |            |             | 8467_1.txt  |
| 5343_8.txt  |            |             |             |
| 7381_8.txt  |            |             |             |
| 8263_9.txt  |            |             |             |
| 8712_8.txt  |            |             |             |
| 9107_7.txt  |            |             |             |
</details>