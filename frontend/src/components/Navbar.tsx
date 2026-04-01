import { Link } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { NavLink as AppNavLink } from "@/components/NavLink";

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navLinks = [
    { href: '/', label: 'Home' },
    { href: '/products', label: 'Products' },
    { href: '/solutions', label: 'Solutions' },
    { href: '/pricing', label: 'Pricing' },
    { href: '/developers', label: 'Developers' },
    { href: '/about', label: 'About' },
    { href: '/contact', label: 'Contact' },
  ];

  return (
    <nav
      className={`fixed top-0 w-full z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-dark/70 backdrop-blur-md border-b border-white/10 shadow-[0_10px_30px_-10px_rgba(0,0,0,0.5)] py-2'
          : 'bg-transparent border-b border-transparent'
      }`}
    >
      <div className="container mx-auto px-4 sm:px-6">
        <div className="flex justify-between items-center h-16 sm:h-20">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <img
              src="/autonomiq-logo.jpeg"
              alt="Autonomiq.AI"
              className="h-10 w-auto max-w-[180px] rounded-md object-contain shadow-[0_8px_30px_-12px_rgba(0,0,0,0.6)] sm:h-12"
            />
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center gap-1">
            {navLinks.map((link, idx) => (
              <motion.div
                key={link.href}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.05 }}
              >
                <AppNavLink
                  to={link.href}
                  end={link.href === "/"}
                  className="px-4 py-2 text-gray-300 hover:text-white rounded-lg transition duration-200 hover:bg-primary/10 text-sm font-medium"
                  activeClassName="bg-primary/10 text-white"
                >
                  {link.label}
                </AppNavLink>
              </motion.div>
            ))}
          </div>

          {/* Right Section */}
          <div className="hidden sm:flex items-center gap-3">
            <Link to="/login">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-4 sm:px-6 py-2 border border-white/10 hover:border-white/30 text-white hover:bg-white/5 rounded-lg font-medium transition duration-200 text-sm"
              >
                Sign In
              </motion.button>
            </Link>
            <Link to="/signup">
              <motion.button
                whileHover={{ scale: 1.05, boxShadow: '0 0 25px rgba(0, 102, 255, 0.45)' }}
                whileTap={{ scale: 0.95 }}
                className="px-4 sm:px-6 py-2 bg-primary hover:bg-primary/90 text-white rounded-lg font-bold transition duration-200 text-sm"
              >
                Get Started
              </motion.button>
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="lg:hidden text-white p-2"
            onClick={() => setIsOpen(!isOpen)}
          >
            <motion.svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              animate={{ rotate: isOpen ? 90 : 0 }}
              transition={{ duration: 0.3 }}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d={isOpen ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16M4 18h16"}
              />
            </motion.svg>
          </button>
        </div>

        {/* Mobile Navigation */}
        <AnimatePresence>
          {isOpen && (
            <motion.div
              className="lg:hidden pb-4 space-y-2 bg-dark-secondary/80 backdrop-blur-xl rounded-xl p-4 mt-2 border border-white/10 shadow-2xl"
              initial={{ opacity: 0, height: 0, y: -20 }}
              animate={{ opacity: 1, height: 'auto', y: 0 }}
              exit={{ opacity: 0, height: 0, y: -20 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
            >
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  to={link.href}
                  className="block px-4 py-3 text-gray-300 hover:text-white hover:bg-primary/10 rounded-lg transition text-sm font-medium"
                  onClick={() => setIsOpen(false)}
                >
                  {link.label}
                </Link>
              ))}
              <div className="flex flex-col gap-2 pt-4 border-t border-white/10">
                <Link to="/login" onClick={() => setIsOpen(false)}>
                  <button className="w-full px-4 py-2 border border-white/10 hover:border-white/30 text-white rounded-lg font-medium transition text-sm">
                    Sign In
                  </button>
                </Link>
                <Link to="/signup" onClick={() => setIsOpen(false)}>
                  <button className="w-full px-4 py-2 bg-primary hover:bg-primary/90 text-white rounded-lg font-bold transition text-sm">
                    Get Started
                  </button>
                </Link>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </nav>
  );
}
