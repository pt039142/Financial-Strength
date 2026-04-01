import SiteFrame from "@/components/SiteFrame";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Link } from "react-router-dom";

const LoginPage = () => {
  return (
    <SiteFrame>
      <section className="relative pt-28 pb-24">
        <div className="container mx-auto grid min-h-[70vh] items-center gap-10 px-6 lg:grid-cols-[0.95fr_1.05fr]">
          <div className="max-w-xl">
            <div className="inline-flex rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-medium uppercase tracking-[0.24em] text-primary">
              Sign in
            </div>
            <h1 className="mt-6 text-4xl font-heading font-bold md:text-5xl">
              Welcome back to your finance automation control room
            </h1>
            <p className="mt-6 text-lg leading-relaxed text-muted-foreground">
              Use your account to review workflows, manage integrations, and keep reconciliation moving.
            </p>
            <div className="mt-8 flex gap-4">
              <Button asChild variant="outline" className="border-border text-foreground hover:bg-secondary">
                <Link to="/signup">Create an account</Link>
              </Button>
              <Button asChild className="bg-primary text-primary-foreground hover:bg-primary/90">
                <Link to="/contact">Talk to sales</Link>
              </Button>
            </div>
          </div>

          <Card className="border-border bg-card shadow-[0_20px_80px_-35px_rgba(0,0,0,0.6)]">
            <CardHeader>
              <CardTitle className="text-2xl">Sign in</CardTitle>
              <CardDescription>Access your workspace and ongoing finance workflows.</CardDescription>
            </CardHeader>
            <CardContent>
              <form className="grid gap-4" onSubmit={(event) => event.preventDefault()}>
                <div className="grid gap-2">
                  <Label htmlFor="login-email">Email</Label>
                  <Input id="login-email" type="email" placeholder="name@company.com" />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="login-password">Password</Label>
                  <Input id="login-password" type="password" placeholder="••••••••" />
                </div>
                <Button type="submit" className="mt-2 bg-primary text-primary-foreground hover:bg-primary/90">
                  Sign in
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </section>
    </SiteFrame>
  );
};

export default LoginPage;
