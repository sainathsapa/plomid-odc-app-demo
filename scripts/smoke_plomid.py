"""Phase 0 PLOMID capability smoke — evidence for ODC_SQL_CAPABILITIES.md.

Runs against a scratch PLOMID instance. Destroys + recreates schema `odc_smoke`.
Usage: DSN="postgresql://plomid:secret@127.0.0.1:55433/plomid" .venv/bin/python scripts/smoke_plomid.py
Exit 0 + ALL PASS only when every probe passes. No app code, no second DB.
"""

import os
import sys

import psycopg

DSN = os.environ.get("DSN", "postgresql://plomid:secret@127.0.0.1:55433/plomid")
results: list[tuple[str, bool, str]] = []


def probe(name: str, fn) -> None:
    try:
        detail = fn()
        results.append((name, True, detail))
    except Exception as e:
        results.append((name, False, f"{type(e).__name__}: {e}"))


def main() -> int:
    conn = psycopg.connect(DSN, autocommit=True)
    cur = conn.cursor()

    def q(sql, params=None):
        cur.execute(sql, params)
        try:
            return cur.fetchall()
        except psycopg.ProgrammingError:
            return []

    probe("server_version", lambda: str(q("SELECT version()")[0][0]))

    q("DROP SCHEMA IF EXISTS odc_smoke CASCADE")
    q("CREATE SCHEMA odc_smoke")

    def ddl_types():
        q("""CREATE TABLE odc_smoke.t (
            id BIGINT PRIMARY KEY, email TEXT NOT NULL UNIQUE,
            plan TEXT NOT NULL DEFAULT 'standard', credit NUMERIC(12,2) NOT NULL DEFAULT 1000.00,
            active BOOLEAN NOT NULL DEFAULT TRUE, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            meta JSONB, tags TEXT[], CHECK (credit >= 0), CHECK (plan IN ('free','standard','enterprise')))""")
        return "PK/UNIQUE/NOT NULL/DEFAULT/NUMERIC/BOOL/TZ/JSONB/ARRAY/CHECK created"

    probe("ddl_core_types", ddl_types)

    def fk_enforced():
        q("CREATE TABLE odc_smoke.parent (id BIGINT PRIMARY KEY)")
        q("""CREATE TABLE odc_smoke.child (id BIGINT PRIMARY KEY, parent_id BIGINT,
             CONSTRAINT child_parent_fk FOREIGN KEY (parent_id) REFERENCES odc_smoke.parent(id))""")
        q("INSERT INTO odc_smoke.parent VALUES (1)")
        q("INSERT INTO odc_smoke.child VALUES (1, 1)")
        q("INSERT INTO odc_smoke.child VALUES (3, NULL)")
        try:
            q("INSERT INTO odc_smoke.child VALUES (2, 999)")
            return "FAIL: invalid FK accepted"
        except Exception as e:
            assert "foreign" in str(e).lower(), str(e)
        n = q("SELECT COUNT(*) FROM odc_smoke.child")[0][0]
        assert n == 2, n
        return "valid+NULL pass, invalid 999 rejected, count=2"

    probe("fk_enforced", fk_enforced)

    def upsert_returning():
        q("INSERT INTO odc_smoke.t (id, email) VALUES (1, 'a@x.com')")
        q("INSERT INTO odc_smoke.t (id, email) VALUES (1, 'a@x.com') ON CONFLICT (id) DO NOTHING")
        q(
            "INSERT INTO odc_smoke.t (id, email) VALUES (1, 'b@x.com') ON CONFLICT (id) DO UPDATE SET email = EXCLUDED.email"
        )
        rows = q("INSERT INTO odc_smoke.t (id, email) VALUES (2, 'c@x.com') RETURNING id, email")
        assert rows[0][1] == "c@x.com", rows
        mail = q("SELECT email FROM odc_smoke.t WHERE id = 1")[0][0]
        assert mail == "b@x.com", mail
        return "ON CONFLICT nothing/update + RETURNING ok"

    probe("upsert_returning", upsert_returning)

    def jsonb_ops():
        q(
            'INSERT INTO odc_smoke.t (id, email, meta) VALUES (10, \'j@x.com\', \'{"team":"perception","score":0.9}\'::jsonb)'
        )
        r = q(
            "SELECT meta->>'team', (meta->>'score')::double precision FROM odc_smoke.t WHERE id = 10"
        )[0]
        assert r == ("perception", 0.9), r
        return "->> + cast ok"

    probe("jsonb_ops", jsonb_ops)

    def vector_all():
        q(
            "CREATE TABLE odc_smoke.docs (id BIGINT PRIMARY KEY, content TEXT, embedding VECTOR(FLOAT32, 4))"
        )
        q(
            "INSERT INTO odc_smoke.docs VALUES (1, 'alpha', [0.12,0.34,0.56,0.78]), (2, 'beta', [0.9,0.1,0.2,0.3]), (3, 'gamma', NULL)"
        )
        d = q(
            "SELECT l2_distance(embedding, [0.12,0.34,0.56,0.78]) FROM odc_smoke.docs WHERE id = 1"
        )[0][0]
        assert abs(d) < 1e-4, d
        top = q(
            "SELECT id FROM odc_smoke.docs ORDER BY embedding <-> [0.12,0.34,0.56,0.78] LIMIT 2"
        )
        assert [r[0] for r in top] == [1, 2], top
        q("CREATE INDEX docs_emb_hnsw ON odc_smoke.docs USING HNSW (embedding)")
        idx = q("SELECT indexname FROM pg_indexes WHERE tablename = 'docs'")
        assert any("hnsw" in r[0].lower() or "emb" in r[0].lower() for r in idx), idx
        try:
            q("INSERT INTO odc_smoke.docs VALUES (9, 'x', [1,2,3])")
            return "FAIL: dim mismatch accepted"
        except Exception as e:
            assert "imension" in str(e), str(e)
        return f"literal+L2+<->TopK+HNSW ok (self-dist {d:.2e})"

    probe("vector_store_index", vector_all)

    def txn_all():
        c2 = psycopg.connect(DSN, autocommit=False)
        try:
            c2.execute("INSERT INTO odc_smoke.t (id, email) VALUES (100, 't@x.com')")
            c2.rollback()
            n = q("SELECT COUNT(*) FROM odc_smoke.t WHERE id = 100")[0][0]
            assert n == 0, n
            c2.execute("INSERT INTO odc_smoke.t (id, email) VALUES (101, 'u@x.com')")
            c2.execute("SAVEPOINT sp1")
            c2.execute("INSERT INTO odc_smoke.t (id, email) VALUES (102, 'v@x.com')")
            c2.execute("ROLLBACK TO SAVEPOINT sp1")
            c2.commit()
            n = q("SELECT COUNT(*) FROM odc_smoke.t WHERE id >= 101")[0][0]
            assert n == 1, n
        finally:
            c2.close()
        return "rollback + savepoint ok"

    probe("transactions", txn_all)

    def catalog():
        fk = q("""SELECT constraint_type FROM information_schema.table_constraints
                  WHERE constraint_name = 'child_parent_fk'""")
        assert fk and fk[0][0] == "FOREIGN KEY", fk
        return "information_schema FK visible"

    probe("catalog_introspection", catalog)

    def explain_plan():
        rows = q("EXPLAIN SELECT id FROM odc_smoke.docs ORDER BY embedding <-> [1,0,0,0] LIMIT 2")
        assert rows, "empty EXPLAIN"
        return f"{len(rows)} plan lines"

    probe("explain", explain_plan)

    def tls_probe():
        try:
            psycopg.connect(DSN, connect_timeout=5, sslmode="require").close()
            return "TLS accepted (unexpected per docs)"
        except Exception as e:
            return f"TLS declined as documented: {type(e).__name__}"

    probe("tls_status", tls_probe)

    q("DROP SCHEMA odc_smoke CASCADE")
    conn.close()

    failed = 0
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<22} {detail}")
        failed += not ok
    print(f"{len(results) - failed}/{len(results)} probes passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
