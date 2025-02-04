# sae.laser
SAE Laser BVDP

## Objectif
pU = Plan passant par le centre optique de la caméra, aligné à la colonne d'index U de la caméra
pV = Plan passant par le centre optique de la caméra, aligné à la ligne d'index U de la caméra
plan vertical = le plan associé à la plateforme verticale
plan horizontal = le plan associé à la plateforme hortizontale

Le but de cette SAE est de reconstruire le modèle 3D d'un objet à partir d'images prises dans un environnement connu à l'aide d'une caméra statique.
Cet environnement est composé d'une plate forme horizontale statique, une plateforme verticale perpendiculaire à la précédente, également statique, et d'un laser qui balaye la scène dans laquelle l'objet est situé.

Les deux plans alignés avec les plateformes horizontale et verticale sont connus. Ainsi, seul le "plan" du laser (qui est dynamique) reste à être déterminé.
Pour déterminer le plan du laser on utilise sa projection sur la scène. La projection du laser sur la plateforme horizontale ne permet pas de déterminer le plan associé au laser, donc la projection du laser sur la plateforme verticale sera utilisée afin de contraindre la solution à un plan unique.

Lorsque le plan du laser est connu, l'intersection du plan vertical, plan horizontal et du plan laser permet de pouvoir associer à chaque pixel une position dans le repère monde. Chacune de ces positions (x,y,z) correspond donc à un point du modèle dense qui est notre objectif. Pour une caméra de résolution u*v, nous devrions avoir en sortie de notre programme au plus u*v points qui nous donneront notre modèle dense. 

Le plan à suivre est donc le suivant : 
1 - Connaitre les équations des plans horizontaux et verticaux
2 - A l'aide des séquences d'images et des paramètres de la caméra, déterminer le plan du laser
3 - Pour chacune des images dans la séquence, pour chacun des points sur l'intersection du plan laser, plan vertical et plan horizontal : associer une position dans le repère monde (ou caméra jsp encore).