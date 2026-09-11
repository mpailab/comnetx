# Аудит замечаний ICDM и план ревизии IEEE Access

Дата аудита: 11 сентября 2026 г.
Версия рукописи: commit `cf29325`
Проверенный PDF: `output/pdf/comnetx-ieee-access.pdf` (16 страниц)

## Итоговая оценка

Текущая журнальная версия существенно сильнее конференционной по постановке
задачи, описанию алгоритма, формальным ограничениям и честности интерпретации
экспериментов. Замечания о порядке обновления иерархии, безусловной
solver-agnostic семантике и отсутствии LD-Leiden в related work в основном
закрыты.

Главный оставшийся риск не является стилистическим: все основные таблицы и
рисунки по-прежнему генерируются из `results/icdm-2026-1`, то есть из измерений,
полученных до исправления иерархии и пространства меток в commit `1061555`.
Внутренний register и measurement README прямо считают исторические smart-mode
результаты с `L >= 2` provisional. Следовательно, почти вся численная база
ComNetX в текущем PDF не может считаться финальной, хотя текст уже описывает
исправленный алгоритм.

До подачи необходимо выбрать одно из двух решений:

1. Перезапустить все сохраняемые claim-bearing эксперименты на исправленном
   коде и полностью регенерировать статью.
2. Удалить из статьи все результаты, которые нельзя подтвердить исправленным
   запуском, и соответственно сузить эмпирические выводы.

Первый вариант предпочтителен. Смешивать старые smart-строки с новым методом
нельзя. Старые full-snapshot строки можно повторно использовать только после
доказанного совпадения source, input, bootstrap, representation, hardware и
timing contract; практически надёжнее выполнить свежие пары.

## Сводка по замечаниям

Статусы:

- **CLOSED** — замечание содержательно закрыто текущим текстом или дизайном;
- **PARTIAL** — overclaim устранён, но исходная научная проблема закрыта не
  полностью;
- **OPEN** — остаётся существенный риск для журнальной подачи;
- **DO NOT PURSUE** — буквальное выполнение замечания ухудшило бы работу или
  потребовало бы недостоверного утверждения.

| Источник | Замечание | Статус | Что уже исправлено | Что остаётся |
|---|---|---|---|---|
| R1.1 | Построение, направление обработки и вложенность иерархии | **CLOSED в тексте; P0 по данным** | Уровни и вложенность определены в `main.tex:392-413`; области фиксируются до обновления (`441-460`); атомы и parent quotients заданы в `464-503`; Algorithm 1 обрабатывает уровни finest-to-coarsest (`552-584`); Proposition 2 устанавливает вложенность (`587-621`). | Новые формулы и код появились после текущих измерений. Их эмпирические результаты необходимо повторить. |
| R1.2, R3 | Слишком сильные solver-agnostic и semantics-preserving claims | **CLOSED** | Используется только `compatible partition-producing detector`; метод прямо назван локальной аппроксимацией (`124-136`); feature-objective preservation отвергается (`254-261`); Proposition 1 ограничена induced scope и atom-constant partitions (`519-543`); boundary mismatch дан в `623-630` и Appendix A. | Полезно лишь кратко перечислить технические условия совместимости: weighted edges, self-loops, один label на quotient vertex и допустимая feature aggregation. Универсальную гарантию для произвольного backend доказывать не следует. |
| R1.3, R3 | Некорректные backend settings и несправедливое cross-backend сравнение | **PARTIAL** | Невалидные DMoN/MAGI/PRGPT/FLMIG/MFC строки удалены; S²CAG явно объявлен fixed-budget interface case, а не matched ranking (`839-850`, `943-955`). | Full и local S²CAG используют разные правила числа кластеров; `T=10` не имеет sensitivity; local state использует Leiden hierarchy. Либо нужен новый matched experiment, либо S²CAG следует оставить только как ограниченную feasibility demonstration без performance ranking. |
| R1.4, R3 | Один удобный real-data window и почти полный initial graph | **PARTIAL** | Конструкции `999:10` и `9:500` описаны точно (`775-801`); добавлены две 500-update trajectories; тип обновлений явно ограничен insertions (`303-305`, `1287-1298`). | Нет независимых или непересекающихся real-data windows. Short backend study остаётся tail-window экспериментом. |
| R1.5, R3 | Повторы не измеряют robustness; недостаточно batch-level evidence | **PARTIAL** | Повторы корректно названы timing repeatability (`780-782`, `902-907`); DSBM использует пять graph seeds; Fig. 3 показывает per-batch `Delta Q` и cumulative time. | Нет per-batch `Delta NMI`, scope/quotient sizes и их связи с drift; нет real-window variation и partition-to-partition agreement. |
| R1.6, R2.W3, R3 | Fallback/refresh только предложен | **OPEN, P1** | Worst case и отсутствие безусловного ускорения описаны (`691-703`); DSBM показывает обе стороны break-even; текст честно сообщает, что controller не реализован (`1265-1272`). | Замечание всех трёх рецензентов остаётся по существу открытым. Полезен только заранее заданный, held-out evaluated controller; быстрый порог, подобранный на тестовых данных, научной ценности не имеет. |
| R2.W1 | Не рассмотрена работа arXiv:2502.18497 | **CLOSED** | LD-Leiden процитирован и содержательно противопоставлен ComNetX (`263-282`); область количественного сравнения объяснена (`854-858`). | Числа внутреннего LD-Leiden campaign не использовать без публикуемой реализации и общего bootstrap/timing protocol. |
| R2.W2 | Abstract выбирает только благоприятный short-horizon результат | **PARTIAL, близко к CLOSED** | Abstract явно разделяет short streams и 500-update result и приводит long-horizon modularity (`75-82`). Results показывают обе long trajectories и NMI degradation (`1162-1205`). | После rerun добавить компактный диапазон по обеим long trajectories или назвать long-horizon NMI gap, чтобы верх статьи полностью отражал drift. |
| R2.W4 | Неработающая ссылка на код | **PARTIAL, P0 перед submission** | Anonymous URL удалён; `https://github.com/mpailab/comnetx` существует. | Default `master` содержит только README/license и не является воспроизводимым artifact статьи; текущий journal commit не опубликован как immutable release/tag. Нужны clean-clone test и стабильная ссылка на release/commit/DOI. |
| R1 Q13/Q17 | Dataset identification и reproducibility | **PARTIAL** | Имена, размеры, timestamp/artificial order, stream construction, hardware, software и clocks описаны (`750-864`, `1274-1298`). | Cora, ACM, Citeseer и PubMed не имеют прямых dataset citations в тексте; отсутствуют raw-file hashes/version, preprocessing, duplicate/self-loop policy, artificial-order seed и единая команда воспроизведения. |
| R3 | Parameter selection для unseen stream | **PARTIAL** | Есть grid `L in {1,2,3,4}`, `r in {0,1,2}`, Pareto frontier и честное утверждение, что общего optimum нет (`1052-1073`). | Нет pilot/held-out правила выбора параметров. Нельзя объявлять `L=3,r=1` рекомендуемым optimum. |
| R3 | Ограниченная scalability evidence | **OPEN, P2** | Есть worst-case analysis, output-sensitive alternative и DSBM с фиксированным `n=100000`, разными update rates и placement. | Нет controlled scaling по `n`, mean community size и relative update volume; крупнейший real graph имеет около 270k vertices. |
| R3 | Качество и temporal consistency по всему потоку | **PARTIAL** | Для 500 updates показан весь `Delta Q_t` trajectory и конечный NMI; DSBM сообщает worst gaps. | Нужны хотя бы per-batch NMI и Local-vs-Full partition agreement; temporal churn/VI полезны, если корректно отделены от snapshot quality. |

## Что не следует делать

1. Не доказывать сохранение objective или поведения произвольного backend: это
   неверно из-за boundary removal, ограниченного atom search space и feature
   aggregation.
2. Не возвращать старые DMoN/MAGI/PRGPT/FLMIG/MFC строки только ради количества
   baseline-методов. Их протоколы смешивали направления, бюджеты, feature modes,
   update counts или некорректные cluster settings.
3. Не выдавать дополнительные повторы одного `999:10` input за robustness.
4. Не строить speedup для LD-Leiden из раздельных campaigns и несовпадающих
   timing intervals.
5. Не подбирать fallback threshold или `(L,r)` на тех же streams/seeds, на
   которых затем показывается результат.
6. Не удалять отрицательные, но валидные результаты. Они поддерживают честное
   утверждение об области применимости метода.

## План доработки

### P0. Восстановить достоверную доказательную базу

1. Заморозить production semantics до измерений:
   - завершить проверку `Optimizer` и launcher;
   - прогнать unit/invariant tests;
   - зафиксировать commit, container digest, dependency versions и input hashes;
   - после старта campaign не менять algorithm/timing code.

2. Выполнить fresh corrected ComNetX campaign:
   - correctness smoke на всех шести `999:10` streams;
   - свежие full/ComNetX пары на всех шести streams;
   - пять повторов на dyn_pubmed и arxivmath использовать только для timing
     repeatability;
   - production-parity mechanism profiles и closure/contraction controls;
   - свежие paired `9:500` trajectories на dyn_pubmed и arxivmath;
   - минимум три, предпочтительно пять, paired DSBM seeds для каждого
     update-rate/placement condition;
   - каждый дополнительный control, который остаётся в статье (topology grid,
     resolution, direction, S²CAG, DF-Leiden), либо перезапустить, либо удалить
     его старые числа из claim-bearing текста.

3. Применить admission gates:
   - zero hierarchy-nesting, namespace-collision и outside-scope partition
     failures;
   - одинаковые paired inputs/bootstrap/graph representation;
   - одинаковый timing contract внутри каждой пары;
   - конечные метрики независимо пересчитаны на полном накопленном графе;
   - неуспешные и неблагоприятные запуски не скрыты.

4. Создать новый immutable bundle в `results/ieee-access-2026-1`, переключить
   `journal/ieee-access/analysis/validate_results.py` с `results/icdm-2026-1`
   на него, затем заново получить все macros, tables и figures. После этого
   построчно сверить Abstract, Results и Conclusion с generated artifacts.

### P1. Закрыть наиболее вероятные повторные замечания

5. Добавить real-window robustness без чрезмерной кампании:
   - выбрать до измерений 3 непересекающихся chronological windows для
     arxivmath и patent;
   - для искусственно упорядоченных streams использовать несколько
     зафиксированных permutation seeds;
   - сообщить по окнам speedup, final/mean/worst `Delta Q`, `Delta NMI` и
     scope/quotient summaries;
   - не смешивать окна с timing repeats одного и того же input.

6. Использовать уже собираемую batch-level телеметрию:
   - построить trajectories для `Delta Q_t`, `Delta NMI_t`,
     `NMI(C_t^local,C_t^full)`, `h_l/n`, `k_l/n`, `e_l/s_t`, `q_l/s_t`;
   - дать median, 95th percentile, worst gap и longest adverse streak;
   - проверить, связан ли drift с ростом scope/quotient или возникает при
     сохранении малой локальной области.

7. Спроектировать fallback только как полноценный эксперимент:
   - runtime branch использует наблюдаемые workload signals и принимает решение
     до backend call; sunk preprocessing cost включается во время;
   - quality branch использует periodic refresh или maximum local-only streak,
     поскольку workload threshold сам по себе не обнаруживает drift;
   - thresholds выбираются на pilot windows/seeds и замораживаются;
   - evaluation проводится на held-out windows/seeds;
   - сравнить Full, Local, periodic-only, workload-only и hybrid по cumulative
     time, worst slowdown/regret, refresh count, `Q`, NMI и partition agreement.
   Если такой дизайн не помещается в ресурсный бюджет, оставить controller как
   явно обозначенную future work, а не добавлять слабый post-hoc threshold.

8. Принять отдельное решение по feature-aware evidence:
   - предпочтительно выполнить один clean matched S²CAG experiment с одинаковым
     заранее заданным cluster-count rule и несколькими training budgets/seeds;
   - backend-matched initialization должна быть отделена от варианта с Leiden
     hierarchy;
   - если это невозможно, перенести количественные S²CAG строки в ограниченную
     interface demonstration и убрать язык, похожий на performance ranking.

9. После новых чисел сбалансировать Abstract:
   - сохранить лучший short-horizon пример;
   - рядом дать диапазон long-horizon speedup и quality gaps по обеим
     trajectories, включая NMI;
   - не использовать слова `preserves quality` без заранее определённого
     критерия.

### P2. Усилить журнальную полноту

10. Добавить компактный controlled DSBM scaling experiment:
    - несколько значений `n` при фиксированной средней степени и relative
      update rate;
    - отдельно варьировать planted community size или число communities;
    - 3-5 seeds;
    - выводить absolute full/local time, peak memory, speedup, scope и quotient
      fractions, а не только time ratio.

11. При наличии бюджета добавить deletion или mixed insertion/deletion DSBM
    control. Если его нет, сохранять явное ограничение insertion-only и не
    обобщать выводы на произвольные dynamic streams.

12. Добавить краткое определение compatible backend: поддерживаемые weighted
    quotient edges и self-loops, один partition label на входную вершину,
    допустимость выбранной feature aggregation и явное правило параметров,
    зависящих от числа clusters.

13. Закрыть dataset provenance:
    - primary citation для каждого набора;
    - version/raw filename и checksum;
    - directedness, symmetrization, duplicate/self-loop handling;
    - features/labels и artificial-order seed;
    - ссылка на download/preprocessing instructions.

### P0 перед загрузкой в IEEE Access

14. Опубликовать immutable artifact:
    - tagged release или archival snapshot с source, configs, accepted records,
      generator и exact environment;
    - проверить воспроизведение tables/figures/PDF из clean clone;
    - заменить moving GitHub URL на stable release/commit/DOI URL;
    - убедиться, что default landing page действительно ведёт к коду статьи.

15. Финальный consistency audit:
    - все числа в тексте происходят из нового validator output;
    - отсутствуют undefined citations/references и ручные численные расхождения;
    - проверены authors, corresponding author, affiliations, funding и ORCID;
    - PDF собран из того же commit, что source archive;
    - визуально проверены все страницы, таблицы и рисунки.

## Текущее состояние PDF

PDF успешно собирается, содержит 16 страниц и 73 цитируемых источника. При
визуальном просмотре всех страниц не обнаружено обрезанного текста,
перекрывающихся объектов или нечитаемых подписей. Fig. 1 и Fig. 2 выглядят
согласованно с текстом. Большое свободное пространство на странице с
биографиями допустимо для текущего no-photo layout. В логе остаются типичные
font/box warnings класса IEEE Access; явных undefined citations или references
нет. Повторную финальную визуальную проверку следует делать только после
замены численных результатов и заморозки текста.

## Критерий готовности

Статья готова к подаче только когда одновременно выполнены четыре условия:

1. Текст метода, production code и claim-bearing measurements относятся к
   одному зафиксированному commit.
2. Все оставшиеся quantitative comparisons проходят общий validator и имеют
   корректную paired provenance.
3. Основные выводы устойчивы хотя бы на нескольких независимых real windows
   или явно ограничены одним окном без generalization claim.
4. Публичный immutable artifact позволяет из clean clone воспроизвести таблицы,
   рисунки и PDF.
