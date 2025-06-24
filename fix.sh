
# 2. Adicione .env ao .gitignore
echo "rag/.env" >> .gitignore
git add .gitignore
git commit -m "Ignorar .env daqui pra frente"

# 3. Instale git-filter-repo
pip install git-filter-repo    # ou via brew, apt, etc.

# 4. Remova o arquivo .env de todo o histórico
git filter-repo --path rag/.env --invert-paths

# 5. Verifique se o .env sumiu do histórico
git log --all -- rag/.env

# 6. Force-push das branches e tags limpas
git push origin --force --all
git push origin --force --tags

