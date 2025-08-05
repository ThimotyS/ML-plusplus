package com.mlquadrat.ml_quadrat_backend.mluser;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface MLUserRepository extends JpaRepository<MLUser, Integer> {
	
	MLUser findByName(String name);
	
}
