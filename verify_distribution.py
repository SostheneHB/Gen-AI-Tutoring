"""Vérifier les octets et le PDF d'une distribution après son extraction."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


def checked_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Chemin non relatif : {name}')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Chemin extérieur au dossier : {name}')
    if not path.is_file():
        raise ValueError(f'Fichier absent : {name}')
    return path


def verify(root, manifest_name):
    manifest = json.loads(checked_path(root, manifest_name).read_text(encoding='utf-8'))
    hashes = manifest['file_sha256']
    for name, expected in hashes.items():
        path = checked_path(root, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Empreinte différente : {name}')
    required = (manifest['main_source'], manifest['pdf'], manifest['bundle_manifest'])
    if any(name not in hashes for name in required):
        raise ValueError('Un fichier principal ne figure pas dans le manifeste.')
    bundle_path = checked_path(root, manifest['bundle_manifest'])
    bundle = json.loads(bundle_path.read_text(encoding='utf-8'))
    for name, expected in bundle['artifact_sha256'].items():
        path = checked_path(bundle_path.parent, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Empreinte du lot différente : {name}')
    for name, expected in bundle['source_sha256'].items():
        if hashlib.sha256(checked_path(root, name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source exécutée différente : {name}')
    reader = PdfReader(checked_path(root, manifest['pdf']))
    if len(reader.pages) != manifest['pdf_pages']:
        raise ValueError('Nombre de pages différent du manifeste.')
    text = ' '.join(' '.join((page.extract_text() or '').split()) for page in reader.pages)
    text = re.sub(r'(?<=\d)\s*\.\s*(?=\d)', '.', text)
    for phrase in manifest['required_pdf_text']:
        if phrase not in text:
            raise ValueError(f'Texte attendu absent du PDF : {phrase}')
    for phrase in manifest['forbidden_pdf_text']:
        if phrase in text:
            raise ValueError(f'Ancien texte encore présent : {phrase}')
    return len(hashes), len(reader.pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--manifest', default='distribution-manifest-2026-09-14.json')
    args = parser.parse_args()
    if args.root.is_absolute():
        parser.error('Utiliser un dossier relatif au répertoire courant.')
    try:
        count, pages = verify(args.root, args.manifest)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f'ÉCHEC : {error}\n')
    print(f'OK : {count} fichiers vérifiés ; PDF de {pages} pages ; lot et sources conformes.')


if __name__ == '__main__':
    main()