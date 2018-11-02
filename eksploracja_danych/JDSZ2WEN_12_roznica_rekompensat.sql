
with kwota_rekompensaty_jezyk as
  (select jezyk, sum(kwota_rekompensaty_oryginalna - kwota_rekompensaty)
   as roznica_rekompensat
   from wnioski
  group by jezyk)

select jezyk, sum(roznica_rekompensat) over (partition by jezyk order by jezyk)
 from kwota_rekompensaty_jezyk
group by jezyk, roznica_rekompensat
order by 2 desc
limit 5;