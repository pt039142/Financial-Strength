import { Link } from 'react-router-dom';

const footerLinks = {
  Product: [
    { label: "Reconcile AI", href: "/solutions" },
    { label: "Invoice AI", href: "/solutions" },
    { label: "Reporting AI", href: "/solutions" },
    { label: "API Platform", href: "/developers" }
  ],
  Company: [
    { label: "About", href: "/about" },
    { label: "Careers", href: "/about" },
    { label: "Blog", href: "/about" },
    { label: "Press", href: "/about" }
  ],
  Resources: [
    { label: "Documentation", href: "/developers" },
    { label: "Partners", href: "/contact" },
    { label: "Support", href: "/contact" },
    { label: "Status", href: "/contact" }
  ],
  Legal: [
    { label: "Privacy", href: "/privacy" },
    { label: "Terms", href: "/terms" },
    { label: "Security", href: "/security" },
    { label: "Compliance", href: "/compliance" }
  ],
};

const Footer = () => {
  return (
    <footer className="border-t border-border py-16">
      <div className="container mx-auto px-6">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-8">
          <div className="col-span-2 md:col-span-1">
            <Link to="/" className="flex items-center gap-2 mb-4">
              <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
                <span className="font-heading text-sm font-bold text-primary-foreground">A</span>
              </div>
              <span className="font-heading text-lg font-bold text-foreground">
                Autonomiq<span className="text-primary">.AI</span>
              </span>
            </Link>
            <p className="text-sm text-muted-foreground leading-relaxed">
              AI-powered finance automation for the modern enterprise.
            </p>
          </div>

          {Object.entries(footerLinks).map(([category, links]) => (
            <div key={category}>
              <h4 className="font-heading font-semibold text-foreground mb-4 text-sm">{category}</h4>
              <ul className="space-y-2">
                {links.map((link) => (
                  <li key={link.label}>
                    <Link to={link.href} className="text-sm text-muted-foreground hover:text-primary transition-colors">
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="border-t border-border mt-12 pt-8 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-xs text-muted-foreground">
            © 2026 Autonomiq.AI. All rights reserved.
          </p>
          <div className="flex gap-6">
            {["Twitter", "LinkedIn", "GitHub"].map((social) => (
              <a key={social} href="#" className="text-xs text-muted-foreground hover:text-primary transition-colors">
                {social}
              </a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
