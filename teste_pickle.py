import pickle

# Carregar arquivo
with open('data/bayes/resultados_bayesianos_completos.pkl', 'rb') as f:
    resultados = pickle.load(f)

print("CHAVES PRINCIPAIS:")
print(list(resultados.keys()))

if 'predicoes_2025' in resultados:
    print("\nESTRUTURA predicoes_2025:")
    print(list(resultados['predicoes_2025'].keys()))
    
    # Mostrar primeiros valores
    for key, value in resultados['predicoes_2025'].items():
        if isinstance(value, dict):
            print(f"\n{key} (dict):")
            print(list(value.keys()))
        elif isinstance(value, list):
            print(f"\n{key} (list): {len(value)} elementos")