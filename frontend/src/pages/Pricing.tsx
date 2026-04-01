import SiteFrame from "@/components/SiteFrame";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Check } from "lucide-react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

const plans = [
  {
    name: "Starter",
    price: "Custom",
    description: "For early teams validating finance automation.",
    features: ["Core workflows", "Email support", "Basic reporting"],
  },
  {
    name: "Growth",
    price: "Custom",
    description: "For teams handling recurring volume and tighter controls.",
    features: ["Advanced reconciliation", "Invoice automation", "Priority support", "Team permissions"],
    featured: true,
  },
  {
    name: "Enterprise",
    price: "Custom",
    description: "For complex, multi-entity organizations and regulated workflows.",
    features: ["Custom integrations", "Dedicated onboarding", "Security review", "SLA support"],
  },
];

const PricingPage = () => {
  return (
    <SiteFrame>
      <section className="relative pt-28 pb-24">
        <div className="container mx-auto px-6">
          <div className="mx-auto max-w-4xl text-center">
            <div className="inline-flex rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-medium uppercase tracking-[0.24em] text-primary">
              Pricing
            </div>
            <h1 className="mt-6 text-4xl font-heading font-bold md:text-5xl">
              Flexible plans for startups, finance teams, and enterprise rollouts
            </h1>
            <p className="mx-auto mt-6 max-w-3xl text-lg leading-relaxed text-muted-foreground">
              We price Autonomiq around scope, integrations, and support requirements so teams can start small and expand without replatforming.
            </p>
            <div className="mt-8">
              <Button asChild size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90 glow-primary px-8">
                <Link to="/contact">Request a custom quote</Link>
              </Button>
            </div>
          </div>

          <div className="mt-16 grid gap-6 lg:grid-cols-3">
            {plans.map((plan, index) => (
              <motion.div
                key={plan.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.08 }}
              >
                <Card className={`h-full border ${plan.featured ? "border-primary/40 bg-primary/5 shadow-[0_20px_80px_-35px_rgba(0,217,255,0.45)]" : "border-border bg-card"}`}>
                  <CardHeader>
                    <CardTitle className="text-2xl">{plan.name}</CardTitle>
                    <CardDescription className="text-sm leading-7">{plan.description}</CardDescription>
                    <div className="pt-4 text-4xl font-heading font-bold text-gradient">{plan.price}</div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {plan.features.map((feature) => (
                      <div key={feature} className="flex items-start gap-3 text-sm text-foreground">
                        <Check className="mt-0.5 h-4 w-4 text-primary" />
                        <span>{feature}</span>
                      </div>
                    ))}
                    <Button asChild className="mt-4 w-full bg-primary text-primary-foreground hover:bg-primary/90">
                      <Link to="/contact">Talk to sales</Link>
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </SiteFrame>
  );
};

export default PricingPage;
