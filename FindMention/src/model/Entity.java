package model;

public class Entity implements Comparable<Entity>{
	String label;
	String uri;
	
	public Entity(String label, String uri){
		this.label = label;
		this.uri = uri;
	}
	
	public String getLabel() {
        return this.label;
    }
    public void setLabel(String label) {
        this.label = label;
    }
    
    public String getUri() {
        return this.uri;
    }
    public void setUri(String uri) {
        this.uri = uri;
    }
    
    public int compareTo(Entity arg0) {
        return this.getLabel().compareTo(arg0.getLabel());
    }
}
