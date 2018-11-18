  select extract (year from data_utworzenia) as rok, jezyk, sum(kwota_rekompensaty_oryginalna - kwota_rekompensaty)
   as roznica_rekompensat
   from wnioski
  where jezyk in ('en','da','pl','sv','pt')
  group by extract (year from data_utworzenia), jezyk
  order by extract (year from data_utworzenia), jezyk
