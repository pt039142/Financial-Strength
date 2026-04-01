import SiteFrame from "@/components/SiteFrame";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Link } from "react-router-dom";

const SignupPage = () => {
  return (
    <SiteFrame>
      <section className="relative pt-28 pb-24">
        <div className="container mx-auto grid min-h-[70vh] items-center gap-10 px-6 lg:grid-cols-[0.95fr_1.05fr]">
          <div className="max-w-xl">
            <div className="inline-flex rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-medium uppercase tracking-[0.24em] text-primary">
              Get started
            </div>
            <h1 className="mt-6 text-4xl font-heading font-bold md:text-5xl">
              Start small, prove value fast, and scale automation across the finance stack
            </h1>
            <p className="mt-6 text-lg leading-relaxed text-muted-foreground">
              Create an account to explore the platform and map the first workflow you want to automate.
            </p>
            <div className="mt-8 flex gap-4">
              <Button asChild variant="outline" className="border-border text-foreground hover:bg-secondary">
                <Link to="/login">Sign in instead</Link>
              </Button>
              <Button asChild className="bg-primary text-primary-foreground hover:bg-primary/90">
                <Link to="/contact">Book a demo</Link>
              </Button>
            </div>
          </div>

          <Card className="border-border bg-card shadow-[0_20px_80px_-35px_rgba(0,0,0,0.6)]">
            <CardHeader>
              <CardTitle className="text-2xl">Create your workspace</CardTitle>
              <CardDescription>Launch your finance automation account in minutes.</CardDescription>
            </CardHeader>
            <CardContent>
              <form className="grid gap-4" onSubmit={(event) => event.preventDefault()}>
                <div className="grid gap-2">
                  <Label htmlFor="signup-name">Full name</Label>
                  <Input id="signup-name" placeholder="Jane Doe" />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="signup-company">Company</Label>
                  <Input id="signup-company" placeholder="Acme Corp" />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="signup-email">Work email</Label>
                  <Input id="signup-email" type="email" placeholder="jane@company.com" />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="signup-password">Password</Label>
                  <Input id="signup-password" type="password" placeholder="••••••••" />
                </div>
                <Button type="submit" className="mt-2 bg-primary text-primary-foreground hover:bg-primary/90">
                  Create account
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </section>
    </SiteFrame>
  );
};

export default SignupPage;
