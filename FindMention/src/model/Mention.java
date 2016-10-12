package model;

import Common.Constant;

/**
 * Created by ethan on 16/3/24.
 */
public class Mention {
    public String label;
    public int pos_start;
    public int pos_end;
    public String prev_context;
    public String after_context;
    public String uris;

    public Mention(){

    }

    public String getLabel(){
        return this.label;
    }

}
