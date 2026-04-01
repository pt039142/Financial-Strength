import { motion } from "framer-motion";
import { ArrowRight, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import heroVisual from "@/assets/hero-visual.jpg";

const BrandHeroSection = () => {
  return (
    <section className="relative min-h-screen flex items-center overflow-hidden pt-20">
      <div className="absolute inset-0 bg-grid opacity-30" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full bg-primary/5 blur-[120px]" />

      <div className="container mx-auto px-6 relative z-10">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <motion.div
            initial={{ opacity: 0, x: -40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="inline-flex items-center gap-3 px-4 py-2 rounded-full border-glow bg-secondary/50 mb-8"
            >
              <img src="/autonomiq-logo.jpeg" alt="" aria-hidden="true" className="h-6 w-6 rounded-sm object-cover" />
              <span className="text-xs font-medium text-muted-foreground">
                Document understanding, fraud detection, and human review
              </span>
            </motion.div>

            <h1 className="text-4xl md:text-5xl lg:text-6xl font-heading font-bold leading-tight mb-6">
              Analyze financial documents with{" "}
              <span className="text-gradient">AI precision</span>
            </h1>

            <p className="text-lg text-muted-foreground max-w-lg mb-10 leading-relaxed">
              Autonomiq turns bank statements, pay stubs, tax forms, invoices, and ledgers into
              audit-ready data, so finance teams can move faster without losing control.
            </p>

            <div className="flex flex-wrap gap-4">
              <Button size="lg" className="bg-primary text-primary-foreground hover:bg-primary/90 glow-primary gap-2 text-base px-8">
                Start Free Trial <ArrowRight size={18} />
              </Button>
              <Button size="lg" variant="outline" className="border-border text-foreground hover:bg-secondary gap-2 text-base px-8">
                <Play size={18} /> Watch Demo
              </Button>
            </div>

            <div className="flex items-center gap-8 mt-12 pt-8 border-t border-border/50">
              {[
                { value: "99+%", label: "Document accuracy" },
                { value: "24/7", label: "Automation" },
                { value: "6+", label: "Verticals" },
              ].map((stat) => (
                <div key={stat.label}>
                  <div className="text-2xl font-heading font-bold text-primary">{stat.value}</div>
                  <div className="text-xs text-muted-foreground">{stat.label}</div>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, delay: 0.5 }}
            className="relative hidden lg:block"
          >
            <div className="relative animate-float">
              <div className="absolute -inset-4 rounded-2xl bg-primary/10 blur-2xl" />
              <img
                src={heroVisual}
                alt="Autonomiq document intelligence platform"
                width={1280}
                height={960}
                className="relative rounded-2xl border-glow"
              />
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default BrandHeroSection;
