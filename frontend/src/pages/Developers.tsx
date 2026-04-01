import SiteFrame from "@/components/SiteFrame";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "react-router-dom";

const endpoints = [
  { method: "POST", path: "/api/v1/auth/login", description: "Issue access tokens for approved users." },
  { method: "POST", path: "/api/v1/invoices/upload", description: "Upload invoices for OCR and processing." },
  { method: "POST", path: "/api/v1/reconciliation/", description: "Run reconciliation against bank and ledger data." },
  { method: "GET", path: "/api/v1/reports/", description: "List generated reporting outputs." },
];

const DevelopersPage = () => {
  return (
    <SiteFrame>
      <section className="relative pt-28 pb-24">
        <div className="container mx-auto px-6">
          <div className="mx-auto max-w-4xl text-center">
            <div className="inline-flex rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-medium uppercase tracking-[0.24em] text-primary">
              Developers
            </div>
            <h1 className="mt-6 text-4xl font-heading font-bold md:text-5xl">
              APIs, webhooks, and system integrations for finance automation teams
            </h1>
            <p className="mx-auto mt-6 max-w-3xl text-lg leading-relaxed text-muted-foreground">
              Autonomiq is designed to plug into the tools your team already uses while keeping the finance workflow centralized and observable.
            </p>
            <div className="mt-8 flex flex-wrap justify-center gap-4">
              <Button asChild size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90 glow-primary px-8">
                <Link to="/contact">Get API access</Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="border-border text-foreground hover:bg-secondary px-8">
                <Link to="/security">Security details</Link>
              </Button>
            </div>
          </div>

          <div className="mt-16 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
            <Card className="border-border bg-card">
              <CardHeader>
                <CardTitle className="text-2xl">Core endpoints</CardTitle>
                <CardDescription>Representative routes from the current backend structure.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {endpoints.map((endpoint) => (
                  <div key={endpoint.path} className="rounded-xl border border-border bg-white/5 p-4">
                    <div className="flex flex-wrap items-center gap-3">
                      <span className="rounded-full bg-primary/15 px-3 py-1 text-xs font-semibold text-primary">{endpoint.method}</span>
                      <code className="text-sm text-foreground">{endpoint.path}</code>
                    </div>
                    <p className="mt-3 text-sm leading-7 text-muted-foreground">{endpoint.description}</p>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card className="border-primary/30 bg-primary/5">
              <CardHeader>
                <CardTitle className="text-2xl">Integration model</CardTitle>
                <CardDescription>How teams should connect to the platform.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 text-sm leading-7 text-muted-foreground">
                <p>Use JWT-protected API calls for user and organization workflows.</p>
                <p>Queue longer-running work like reconciliation and invoice processing through the backend task layer.</p>
                <p>Store documents in object storage and keep finance data in PostgreSQL for traceability.</p>
                <p>Use Redis-backed Celery workers for async jobs and background synchronization.</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>
    </SiteFrame>
  );
};

export default DevelopersPage;
