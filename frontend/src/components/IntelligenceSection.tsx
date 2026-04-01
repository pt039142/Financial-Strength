import { motion } from "framer-motion";
import { Bot, FileCheck2, ShieldAlert, Sparkles, Workflow, LineChart, GitBranch, ScanText } from "lucide-react";

const features = [
  {
    icon: ScanText,
    title: "Document understanding",
    description: "Specialized capture for bank statements, pay stubs, tax forms, invoices, and supporting evidence.",
  },
  {
    icon: ShieldAlert,
    title: "Fraud detection",
    description: "Detect tampering, anomalies, mismatches, and risk signals before bad data reaches the workflow.",
  },
  {
    icon: Workflow,
    title: "Model orchestration",
    description: "Route tasks to the best model or workflow step for the job at hand, instead of one-size-fits-all processing.",
  },
  {
    icon: FileCheck2,
    title: "Human-in-the-loop review",
    description: "Escalate low-confidence items to reviewers so edge cases are resolved with control and accountability.",
  },
  {
    icon: LineChart,
    title: "Cash flow analysis",
    description: "Measure revenue, liabilities, and debt capacity from source documents and transaction data.",
  },
  {
    icon: Bot,
    title: "Income calculations",
    description: "Evaluate income across borrower, contractor, or applicant types with consistent rules and outputs.",
  },
  {
    icon: GitBranch,
    title: "Vertical solutions",
    description: "Support lending, mortgage, tax, legal, tenant screening, and other document-heavy workflows.",
  },
  {
    icon: Sparkles,
    title: "Audit-ready outputs",
    description: "Produce structured, reviewable results designed for downstream systems and compliance teams.",
  },
];

const IntelligenceSection = () => {
  return (
    <section id="platform" className="py-24 relative">
      <div className="absolute inset-0 bg-grid opacity-10" />
      <div className="container mx-auto px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="text-xs font-medium text-primary uppercase tracking-widest">Platform</span>
          <h2 className="text-3xl md:text-4xl font-heading font-bold mt-4 mb-4">
            Document and <span className="text-gradient">Decision Intelligence</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Built for high-volume financial workflows where accuracy, verification, and explainability all matter.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-[1.1fr_0.9fr] gap-8 items-start">
          <motion.div
            variants={{
              hidden: {},
              show: { transition: { staggerChildren: 0.08 } },
            }}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="grid md:grid-cols-2 gap-6"
          >
            {features.map((feature) => (
              <motion.div
                key={feature.title}
                variants={{
                  hidden: { opacity: 0, y: 24 },
                  show: { opacity: 1, y: 0, transition: { duration: 0.5 } },
                }}
                className="group p-8 rounded-xl bg-card border border-border hover:border-primary/30 transition-all duration-500 hover:glow-primary"
              >
                <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-6 group-hover:bg-primary/20 transition-colors">
                  <feature.icon className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-lg font-heading font-semibold mb-3 text-foreground">{feature.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 24 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur"
          >
            <div className="inline-flex items-center gap-3 rounded-full border border-white/10 bg-black/20 px-4 py-2 text-xs uppercase tracking-[0.24em] text-primary">
              <img src="/autonomiq-logo.jpeg" alt="" aria-hidden="true" className="h-5 w-5 rounded-sm object-cover" />
              Built for trust
            </div>
            <h3 className="mt-6 text-2xl font-heading font-bold">What this platform is optimized for</h3>
            <p className="mt-4 text-sm leading-7 text-muted-foreground">
              We designed Autonomiq around the same core themes the best document-intelligence platforms use:
              precise document understanding, risk-aware verification, and structured outputs that can flow into
              lending, accounting, and compliance systems.
            </p>

            <div className="mt-8 space-y-4">
              {[
                "99+% document handling precision",
                "Maker-checker workflows for edge cases",
                "Industry-specific decision support",
                "API-first integration for downstream systems",
              ].map((item) => (
                <div key={item} className="rounded-xl border border-border bg-card/70 px-4 py-3 text-sm text-foreground">
                  {item}
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default IntelligenceSection;
