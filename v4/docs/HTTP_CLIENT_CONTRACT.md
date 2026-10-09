# SEC-1 / SEC-2 — contrat du transport HTTP local

Statut : serveur implémenté dans la candidate ; adaptation des clients par Claude
(SEC-2), puis qualification et autorisation de bascule. Aucun service de production
n'est modifié par SEC-1. SDK qualifié : MCP Python 1.26.0 ; minimum du paquet relevé
à cette version pour `TransportSecuritySettings` et le gestionnaire HTTP.

## Configuration serveur obligatoire

`MEMORY_V4_TOKEN_FILE` désigne un fichier ordinaire appartenant à l'utilisateur du
serveur, permissions exactement **0600**, sans lien symbolique/physique. Il contient
un jeton URL-safe ASCII de 43 à 256 caractères, avec au plus un saut de ligne final.
Créer le jeton avec `secrets.token_urlsafe(32)` (256 bits), sans jamais l'imprimer.
Le serveur refuse de démarrer si cette configuration manque ou est invalide, avant
de charger le moteur, même avec `--warm`. Aucun mode HTTP sans authentification.
La variable désigne un **chemin**, jamais un secret. Aucun secret en argument CLI,
URL, fichier Git ou sortie de diagnostic.

Le bind HTTP ALPHA accepte seulement `127.0.0.1`, `localhost`, `::1`. Le port par
défaut reste 8765. Pour le port configuré, les seuls Host admis sont exactement
`127.0.0.1:<port>`, `localhost:<port>`, `[::1]:<port>`. Les seuls Origin admis sont
les trois origines `http://` correspondantes, sans joker ni autre port. Une Origin
absente est admise pour les clients natifs ; elle ne dispense pas du jeton.
Origin `null`, domaines tiers, doublons Host/Origin/Authorization : refus.

## Configuration cliente à réaliser par Claude (SEC-2)

Toutes les requêtes HTTP, **y compris `/health`**, nécessitent :

```text
Authorization: Bearer <valeur lue depuis le fichier 0600>
```

Le client/relais lit le même fichier protégé. Avec le SDK 1.26, utiliser
`httpx.AsyncClient(headers={"Authorization": "Bearer " + token})` et transmettre
ce client à `streamable_http_client(url, http_client=client)`. Fermer le client HTTP
après la session. L'ancien `streamablehttp_client(url, headers=...)` accepte également
les en-têtes mais est déprécié : privilégier l'API actuelle.

Relais stdio : lire le fichier au démarrage et à la reconnexion ; ne jamais transmettre
le secret sur stdout/stderr. Configs : enregistrer seulement le chemin du fichier dans
l'environnement du relais. Pour les clients HTTP directs, vérifier leur mécanisme
d'injection des en-têtes ; s'il impose un secret dans un fichier non protégé ou suivi
par Git, préférer le relais plutôt que recopier le jeton dans la configuration.
Adapter aussi sondes, scripts de santé et procédure de redémarrage.

Réponses : 401 + `WWW-Authenticate: Bearer` (jeton absent/invalide), 421 (Host rejeté),
403 (Origin rejetée), 400 (en-têtes ambigus ou POST non JSON). Ces contrôles précèdent
le routage, les outils et toute mutation. Un client ne doit pas réessayer une mutation
avec un nouvel identifiant après une coupure : résultat incertain, contrat §5 du plan.
Il n'existe aucune exception anonyme pour `/health` ou `/ready`.
Le [contrat readiness](READINESS.md) décrit les états et le commit du build à fournir.

## Rotation / révocation

Le serveur charge le jeton une seule fois au démarrage. Remplacer atomiquement son
fichier par un nouveau fichier 0600, puis redémarrer le serveur et reconnecter les
clients : c'est le redémarrage qui révoque effectivement l'ancien jeton. Les requêtes
en vol peuvent avoir été appliquées ; conserver leurs request_id. L'ALPHA n'expose
pas d'API de rotation à chaud ni de période d'acceptation de deux jetons.
Le jeton ne protège pas contre un processus compromis ayant les mêmes droits utilisateur.

## Preuves avant bascule

```sh
python -m pytest v4/tests/test_http_security.py -q
python -m pytest v4/tests/test_server_http_e2e.py -q
```

Les tests ASGI passent par la véritable application, le gestionnaire MCP et le moteur
`hash` sur un magasin temporaire : initialisation, écriture, rappel et rejeu identifié ;
Host/Origin/jeton invalides n'altèrent aucun fichier du magasin. Ils ne remplacent pas
les tests TCP Uvicorn + clients MCP, ni les parcours des applications Claude/Codex.
Les tests TCP doivent être exécutés hors d'un sandbox interdisant le bind loopback,
sur un magasin synthétique temporaire et un port libre, jamais sur la production.
SEC-2 exige enfin une preuve datée après redémarrage des deux clients, sans divulgation
du jeton. La bascule demeure conditionnée à ces preuves et à la revue de Claude.
