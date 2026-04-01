import SiteFrame from "@/components/SiteFrame";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { motion } from "framer-motion";

const ContactPage = () => {
  return (
    <SiteFrame>
      <section className="relative pt-28 pb-24">
        <div className="container mx-auto px-6">
          <div className="mx-auto max-w-4xl text-center">
            <div className="inline-flex rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-medium uppercase tracking-[0.24em] text-primary">
              Contact
            </div>
            <h1 className="mt-6 text-4xl font-heading font-bold md:text-5xl">
              Book a demo and see the platform in a real finance workflow
            </h1>
            <p className="mx-auto mt-6 max-w-3xl text-lg leading-relaxed text-muted-foreground">
              Tell us how your team works today and we’ll map the right rollout path, from one workflow to a full finance automation stack.
            </p>
          </div>

          <div className="mt-16 grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
            <Card className="border-border bg-card">
              <CardHeader>
                <CardTitle className="text-2xl">Start the conversation</CardTitle>
                <CardDescription>We typically respond within one business day.</CardDescription>
              </CardHeader>
              <CardContent>
                <form className="grid gap-4" onSubmit={(event) => event.preventDefault()}>
                  <div className="grid gap-2">
                    <Label htmlFor="name">Full name</Label>
                    <Input id="name" placeholder="Jane Doe" />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="email">Work email</Label>
                    <Input id="email" type="email" placeholder="jane@company.com" />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="company">Company</Label>
                    <Input id="company" placeholder="Acme Corp" />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="message">What do you want to automate?</Label>
                    <Textarea id="message" placeholder="Tell us about reconciliation, invoices, reporting, or integrations..." rows={5} />
                  </div>
                  <Button type="submit" className="mt-2 bg-primary text-primary-foreground hover:bg-primary/90">
                    Request a demo
                  </Button>
                </form>
              </CardContent>
            </Card>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="space-y-6"
            >
              <Card className="border-primary/30 bg-primary/5">
                <CardHeader>
                  <CardTitle>What we cover in a demo</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm leading-7 text-muted-foreground">
                  <p>Workflow fit and rollout sequencing.</p>
                  <p>Security, compliance, and access boundaries.</p>
                  <p>Integration points with your accounting stack.</p>
                </CardContent>
              </Card>
              <Card className="border-border bg-card">
                <CardHeader>
                  <CardTitle>Direct contact</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm leading-7 text-muted-foreground">
                  <p>Email: hello@autonomiq.ai</p>
                  <p>Sales: +91 00000 00000</p>
                  <p>Support: support@autonomiq.ai</p>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </div>
      </section>
    </SiteFrame>
  );
};

export default ContactPage;
