# Prova-Visao-computacional

Aluno: José Vitor Gonçalves
RGM: 30654114

Olá, professor, espero que esteja bem, segue descritivo dos meus códigos da prova.
Eu desenvolvi os códigos no Colab, estou entregando por aqui para melhor organização e inclusive disponibilização das imagens usadas para o treinmaneto.
Os links para os projetos no Colab estão abaixo:

Se entrar nos links não vai conseguir rodar, pois está sem as imagens, que só ficam armazenadas na sessão que está ativa, entretanto pode ver o código já executado e os resultados obtidos.

Link Colab Treinamento:
https://colab.research.google.com/drive/1BoENR3P7D_YW5u8MQpjCMa1cr6pcSm1l?authuser=4#scrollTo=A0-SuDIxw1KT

Link Colab Pré-processamento:
https://colab.research.google.com/drive/1UFCJjGhd500x09ugh8Ej0wpjmY2uKd56?authuser=4

Vamos as defesas dos códigos:

Bom, a minha ideia inicial foi a classificação de dois tipos de carros diferentes, inicialmente sedans ou caminhonetes, achei que pelo contraste dos modelos poderia obter resultados interessantes.
O desenvolvimento foi em volta desse tema, então encontrei um bom dataset que contém várias imagens separadas por tipo de carro.
Dataset: Vehicle Type Image Dataset (Version 1): VTID1
Link: https://data.mendeley.com/datasets/r7bthvstxw/2

Como dito, o dataset contém várias imagens e são padronizadas e separadas em pastas de acordo com o tipo do carro.

Primeira coisa, o código vê as imagens no caminho determinado, e as classifica, 0 para hatchs e 1 para suv, de acordo com as pastas que estão separadas.
E após isso é feito um redimensionamento para que todas as imagens estejam padronizadas por tamanho.

A fins de teste, deixei um trecho que printa as 5 primeiras fotos de cada pasta, para ver se estão sendo lidas corretamente e para avaliar a diferença entre as fotos.

Após, as imagens são normalizadas e é divida a quantidade de treino e de teste, sendo 80% treino e 20% teste, como solicitado.

E então, é feito o One Hot Enconding que:
Transforma os rótulos das classes (0, 1, etc.) em vetores binários chamados one-hot.
Isso é necessário quando usamos categorical_crossentropy como função de perda, pois o modelo espera que os rótulos estejam nesse formato.

Estamos usando o categorical_crossentropy pois foi o parâmetro que entregou melhor resultado nos testes.

Então, definimos as Class Names (hatch, suv) e fazemos o treinamento da rede, os parâmetros usados são os mesmos vistos em aula, entregaram um bom resultado.

Em seguida fazemos todos os testes de acurácia e predição.

Após os testes de acurácia entramos com novas imagens para a rede testar, entretando isso foi o problema, fiz várias vezes e sempre os testes de acurácia mostravam um bom resultado da rede, mas ao testar com imagens novas a classificação na maioria das vezes errada, inclusive nessa versão dos códigos e testes, mencionei lá em cima que iria classificar entre sedans e caminhonetes, mas não consegui um resultado satisfatório, todas as novas imagens eram classificadas como caminhonetes, então fui testando modelos diferentes e cheguei nesse teste entre Hatchs e Suvs, que pelo menos entregou alguns carros classificados corretamente. Talvez essa imprecisão tenha sido causada pelo baixo numero de imagens usadas no treinamento ou até da forma que essas imagens são, muito "irreais".


