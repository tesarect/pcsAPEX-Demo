{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python310
    python310Packages.django
    python310Packages.pillow
    python310Packages.tensorflow
    python310Packages.scikit-learn
    python310Packages.numpy
    python310Packages.opencv4
    python310Packages.psycopg2
    python310Packages.pip
    python310Packages.virtualenv
    postgresql
  ];

  shellHook = ''
    # Create a virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
      echo "Creating virtual environment..."
      virtualenv venv
    fi
    
    # Activate the virtual environment
    source venv/bin/activate
    
    # Install dependencies if needed
    if [ ! -f ".requirements-installed" ]; then
      echo "Installing dependencies..."
      pip install -r requirements.txt
      touch .requirements-installed
    fi
    
    echo "Python environment ready!"
  '';
}
