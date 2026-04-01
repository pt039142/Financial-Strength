import { motion } from "framer-motion";
import { Bot, FileCheck, BarChart3, Shield, Zap, Globe } from "lucide-react";

const features = [
  {
    icon: FileCheck,
    title: "Reconcile AI",
    description: "Automated bank reconciliation with 99.7% accuracy. Match thousands of transactions in seconds.",
  },
  {
    icon: Bot,
    title: "Invoice Processing AI",
    description: "Extract, validate, and process invoices automatically. Eliminate manual data entry entirely.",
  },
  {
    icon: BarChart3,
    title: "Financial Reporting AI",
    description: "Generate compliance-ready financial reports in real time with intelligent data aggregation.",
  },
  {
    icon: Shield,
    title: "Compliance Engine",
    description: "Stay audit-ready with automated IFRS/GAAP compliance checks and risk assessment.",
  },
  {
    icon: Zap,
    title: "API Platform",
    description: "Plug into your existing stack with powerful APIs for invoice-extract, reconcile, and audit-risk.",
  },
  {
    icon: Globe,
    title: "Multi-Currency",
    description: "Global-ready with multi-currency accounting, cross-border compliance, and regional tax support.",
  },
];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1 } },
};

const item = {
  hidden: { opacity: 0, y: 30 },
  show: { opacity: 1, y: 0, transition: { duration: 0.6 } },
};

const FeaturesSection = () => {
  return (
    <section id="products" className="py-24 relative">
      <div className="absolute inset-0 bg-grid opacity-10" />
      <div className="container mx-auto px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="text-xs font-medium text-primary uppercase tracking-widest">Products</span>
          <h2 className="text-3xl md:text-4xl font-heading font-bold mt-4 mb-4">
            The Complete <span className="text-gradient">Finance Automation</span> Suite
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            From reconciliation to reporting, our AI agents handle the heavy lifting across your entire finance workflow.
          </p>
        </motion.div>

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {features.map((feature) => (
            <motion.div
              key={feature.title}
              variants={item}
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
      </div>
    </section>
  );
};

export default FeaturesSection;
