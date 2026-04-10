/**
* @author: Mateus H. Fulan
* @description: Script para download da coleção de classificações de uso e cobertura da terra
+               (coleção 10) do MapBiomas usando uma 'regiao_de_interesse' no formato FeatureCollection.
*/

var extrair_anos = [
    1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992,
    1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
    2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
    2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024
  ];

var regiao_de_interesse = ee.FeatureCollection("projects/ee-mateusfulan-research/assets/fiat-firms/cerrado-clip-sp");
var mapbiomas = ee.Image("projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_coverage_v2");

var processar_ano = function(ano) {
  var clip_ano = mapbiomas
                 .select("classification_" + ano)
                 .clip(regiao_de_interesse);

  Export.image.toDrive({
      image: clip_ano,
      description: "Exportar_" + ano,
      folder: "SAIDA_MAPBIOMAS_ANOS",
      fileNamePrefix: "classificacao_" + ano,
      region:regiao_de_interesse,
      scale: 30,
      maxPixels:1e13,
  });
};

extrair_anos.forEach(processar_ano);
