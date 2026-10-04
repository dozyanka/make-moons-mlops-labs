# Настройка GitHub перед сдачей

1. Создать публичный или доступный преподавателю репозиторий и отправить содержимое архива.
2. `Settings -> Branches -> Add branch protection rule` для `main`:
   - Require a pull request before merging;
   - Require status checks to pass before merging;
   - выбрать workflow `CI / test`;
   - Block force pushes;
   - Require conversation resolution.
3. Включить Actions.
4. Сделать хотя бы один учебный PR через отдельную ветку, дождаться зелёного CI и слить его.
5. Ссылку на репозиторий вставить в титульные части отчётов перед сдачей.

Защита ветки — серверная настройка GitHub и не может быть закодирована внутри ZIP-файла.
