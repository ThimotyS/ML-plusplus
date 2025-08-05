package com.mlquadrat.ml_quadrat_backend.config;

import java.util.HashMap;

import java.util.Map;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import response.MLQuadratError;


@RestControllerAdvice
public class GlobalExceptionHandler {
	
	@ExceptionHandler(MLQuadratError.class)
	public ResponseEntity<Map<String,String>> handleMLQuadratError(MLQuadratError ex){
		Map<String,String> response = new HashMap();
		response.put("message", "There's an error regarding ML2's functionality: " + ex.getMessage());
		return new ResponseEntity<>(response, HttpStatus.INTERNAL_SERVER_ERROR);
	}
	
	@ExceptionHandler(Exception.class) 
	public ResponseEntity<Map<String,String>> handleOtherError(Exception ex){
		Map<String,String> response = new HashMap();
		response.put("message", "There's a server error: " + ex);
		return new ResponseEntity<>(response, HttpStatus.INTERNAL_SERVER_ERROR);
	}
}
